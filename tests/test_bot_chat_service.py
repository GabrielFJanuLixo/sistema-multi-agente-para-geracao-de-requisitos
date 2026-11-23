"""Integração da API do chat com banco de teste, grafo real e Gemini simulado."""

import json
from collections import deque
from http import HTTPStatus

import pytest
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError

from multi_agent_chat.app import app
from multi_agent_chat.chatbot.estados.schemas import Entrevista
from multi_agent_chat.core.chatbot import get_model_factory
from tests.bot_helpers import criar_especificacao, criar_modelos


@pytest.fixture
def modelos_chat(client):
    # O client já prepara o banco isolado e o checkpoint da aplicação.
    modelos = criar_modelos()

    def fabrica(agent):
        return modelos[agent.role.value]

    app.dependency_overrides[get_model_factory] = lambda: fabrica
    yield modelos
    app.dependency_overrides.pop(get_model_factory, None)


def _eventos_sse(texto):
    eventos = []
    for bloco in texto.replace('\r\n', '\n').split('\n\n'):
        nome = None
        dados = []
        for linha in bloco.splitlines():
            if linha.startswith('event:'):
                nome = linha.partition(':')[2].strip()
            elif linha.startswith('data:'):
                dados.append(linha.partition(':')[2].lstrip())
        if nome is not None and dados:
            eventos.append((nome, json.loads('\n'.join(dados))))
    return eventos


@pytest.mark.usefixtures('modelos_chat')
def test_chat_salva_mensagens_status_e_especificacao_consultavel(client, conversation, token):
    base = f'/conversations/{conversation.id}'
    headers = {'Authorization': f'Bearer {token}'}

    resposta = client.post(base + '/chat', headers=headers, json={'content': 'Quero acompanhar pedidos.'})

    assert resposta.status_code == HTTPStatus.OK
    dados = resposta.json()
    assert dados['status'] == 'completed'
    assert dados['user_message']['user_id'] == conversation.user_id
    assert dados['user_message']['agent_id'] is None
    assert dados['assistant_message']['agent_id'] is not None
    assert dados['assistant_message']['user_id'] is None
    mensagens = client.get(base + '/messages/', headers=headers).json()
    assert [mensagem['id'] for mensagem in mensagens] == [
        dados['user_message']['id'],
        dados['assistant_message']['id'],
    ]
    assert client.get(base, headers=headers).json()['status'] == 'completed'
    especificacao = client.get(base + '/specification', headers=headers)
    assert especificacao.status_code == HTTPStatus.OK
    assert especificacao.json()['specification'] == dados['specification']
    markdown = client.get(base + '/specification/markdown', headers=headers)
    assert markdown.status_code == HTTPStatus.OK
    assert 'RF-001' in markdown.text
    assert 'Um pedido existente apresenta seu status.' in markdown.text


def test_chat_retoma_pergunta_e_acrescenta_mensagens_sem_perder_historico(client, conversation, token, modelos_chat):
    especificacao = criar_especificacao()
    modelos_chat['interviewer'].respostas[Entrevista] = deque([
        Entrevista(especificacao=especificacao, perguntas=['Quem consulta?']),
        Entrevista(especificacao=especificacao),
    ])
    base = f'/conversations/{conversation.id}'
    headers = {'Authorization': f'Bearer {token}'}

    inicio = client.post(base + '/chat', headers=headers, json={'content': 'Quero acompanhar pedidos.'})
    assert inicio.status_code == HTTPStatus.OK
    assert inicio.json()['status'] == 'waiting_user'
    assert inicio.json()['question'] == 'Quem consulta?'
    assert inicio.json()['question_index'] == 1

    final = client.post(base + '/chat', headers=headers, json={'content': 'Somente o titular.'})
    assert final.status_code == HTTPStatus.OK
    assert final.json()['status'] == 'completed'
    assert final.json()['specification']['entrevistas'][0]['respostas'] == ['Somente o titular.']
    mensagens = client.get(base + '/messages/', headers=headers).json()
    assert [mensagem['id'] for mensagem in mensagens] == [
        inicio.json()['user_message']['id'],
        inicio.json()['assistant_message']['id'],
        final.json()['user_message']['id'],
        final.json()['assistant_message']['id'],
    ]
    assert mensagens[0]['content'] == 'Quero acompanhar pedidos.'
    assert mensagens[2]['content'] == 'Somente o titular.'


@pytest.mark.usefixtures('modelos_chat')
def test_conversa_concluida_rejeita_novo_turno_sem_salvar_mensagem(client, conversation, token):
    base = f'/conversations/{conversation.id}'
    headers = {'Authorization': f'Bearer {token}'}
    inicial = client.post(base + '/chat', headers=headers, json={'content': 'Quero acompanhar pedidos.'})
    assert inicial.status_code == HTTPStatus.OK
    mensagens_anteriores = client.get(base + '/messages/', headers=headers).json()

    resposta = client.post(base + '/chat', headers=headers, json={'content': 'Outra descrição.'})

    assert resposta.status_code == HTTPStatus.CONFLICT
    assert resposta.json()['detail'] == 'Conversation is already completed'
    assert client.get(base + '/messages/', headers=headers).json() == mensagens_anteriores


@pytest.mark.usefixtures('modelos_chat')
@pytest.mark.parametrize('endpoint', ['chat', 'chat/stream', 'specification', 'specification/markdown'])
def test_rotas_do_bot_exigem_login_e_propriedade_da_conversa(client, conversation, other_token, endpoint):
    metodo = 'post' if endpoint.startswith('chat') else 'get'
    url = f'/conversations/{conversation.id}/{endpoint}'
    corpo = {'json': {'content': 'Quero acompanhar pedidos.'}} if metodo == 'post' else {}

    sem_login = client.request(metodo, url, **corpo)
    outro_usuario = client.request(metodo, url, headers={'Authorization': f'Bearer {other_token}'}, **corpo)

    assert sem_login.status_code == HTTPStatus.UNAUTHORIZED
    assert outro_usuario.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.parametrize('endpoint', ['chat', 'chat/stream'])
def test_descricao_inicial_vazia_na_api_nao_chama_modelo_nem_salva_mensagens(client, conversation, token, modelos_chat, endpoint):
    base = f'/conversations/{conversation.id}'
    headers = {'Authorization': f'Bearer {token}'}

    resposta = client.post(base + '/' + endpoint, headers=headers, json={'content': ' \t '})

    assert resposta.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert all(not modelo.chamadas for modelo in modelos_chat.values())
    assert client.get(base + '/messages/', headers=headers).json() == []


@pytest.mark.parametrize(
    'cenario',
    [(429, HTTPStatus.TOO_MANY_REQUESTS), (503, HTTPStatus.SERVICE_UNAVAILABLE), (500, HTTPStatus.BAD_GATEWAY)],
)
def test_falha_do_gemini_retorna_status_adequado_sem_salvar_turno(client, conversation, token, modelos_chat, cenario):
    codigo, esperado = cenario
    erro = (
        GoogleRateLimitError('Limite simulado.')
        if codigo == HTTPStatus.TOO_MANY_REQUESTS
        else GoogleAPIError(codigo, {'error': {'message': 'Falha simulada.'}})
    )
    modelos_chat['interviewer'].respostas[Entrevista] = deque([erro])
    base = f'/conversations/{conversation.id}'
    headers = {'Authorization': f'Bearer {token}'}

    resposta = client.post(base + '/chat', headers=headers, json={'content': 'Quero acompanhar pedidos.'})

    assert resposta.status_code == esperado
    assert client.get(base + '/messages/', headers=headers).json() == []
    assert client.get(base, headers=headers).json()['status'] == 'idle'


@pytest.mark.usefixtures('modelos_chat')
def test_stream_da_api_emite_progresso_e_conclusao_com_especificacao(client, conversation, token):
    base = f'/conversations/{conversation.id}'
    headers = {'Authorization': f'Bearer {token}'}

    resposta = client.post(base + '/chat/stream', headers=headers, json={'content': 'Quero acompanhar pedidos.'})

    assert resposta.status_code == HTTPStatus.OK
    assert resposta.headers['content-type'].startswith('text/event-stream')
    eventos = _eventos_sse(resposta.text)
    assert eventos[0] == ('status', {'mensagem': 'Processando...'})
    assert eventos[-1][0] == 'done'
    assert eventos[-1][1]['status'] == 'completed'
    assert eventos[-1][1]['specification']['especificacao']['requisitos'][0]['id'] == 'RF-001'
    assert not any(nome == 'error' for nome, _ in eventos)
    mensagens = client.get(base + '/messages/', headers=headers).json()
    conclusao = eventos[-1][1]
    assert [mensagem['id'] for mensagem in mensagens] == [
        conclusao['user_message']['id'],
        conclusao['assistant_message']['id'],
    ]


def test_falha_no_stream_emite_evento_de_erro_sem_conclusao_ou_turno_salvo(client, conversation, token, modelos_chat):
    modelos_chat['interviewer'].respostas[Entrevista] = deque([
        GoogleAPIError(503, {'error': {'message': 'Indisponibilidade simulada.'}}),
    ])
    base = f'/conversations/{conversation.id}'
    headers = {'Authorization': f'Bearer {token}'}

    resposta = client.post(base + '/chat/stream', headers=headers, json={'content': 'Quero acompanhar pedidos.'})

    # No streaming, os cabeçalhos HTTP já foram enviados: a falha é um evento SSE.
    assert resposta.status_code == HTTPStatus.OK
    eventos = _eventos_sse(resposta.text)
    assert eventos[-1][0] == 'error'
    assert not any(nome == 'done' for nome, _ in eventos)
    assert client.get(base + '/messages/', headers=headers).json() == []


@pytest.mark.usefixtures('modelos_chat')
@pytest.mark.parametrize('endpoint', ['specification', 'specification/markdown'])
def test_consulta_de_especificacao_antes_da_geracao_retorna_404(client, conversation, token, endpoint):
    resposta = client.get(
        f'/conversations/{conversation.id}/{endpoint}',
        headers={'Authorization': f'Bearer {token}'},
    )

    assert resposta.status_code == HTTPStatus.NOT_FOUND
    assert resposta.json()['detail'] == 'Specification not generated yet'
