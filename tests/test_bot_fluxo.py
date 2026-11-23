"""Integração do LangGraph real com respostas locais em vez de chamadas de IA."""

import pytest

from multi_agent_chat.chatbot.controle.orquestrador import (
    EspecialistaConfig,
    Orquestrador,
    OrquestradorConfig,
    create_checkpointer,
)
from multi_agent_chat.chatbot.estados.schemas import Consolidacao, Entrevista, Parecer
from tests.bot_helpers import criar_especificacao, criar_modelos, criar_parecer


@pytest.mark.asyncio
@pytest.mark.parametrize('descricao', ['', '   ', '\n\t'])
async def test_descricao_vazia_e_rejeitada_sem_chamar_os_agentes(descricao):
    modelos = criar_modelos()
    orquestrador = Orquestrador(modelos)

    with pytest.raises(ValueError, match='descrição do projeto não pode estar vazia'):
        await orquestrador.iniciar('vazio', descricao)

    assert all(not modelo.chamadas for modelo in modelos.values())


@pytest.mark.asyncio
async def test_fluxo_sem_perguntas_aprova_apos_tres_avaliacoes():
    modelos = criar_modelos()
    orquestrador = Orquestrador(modelos)

    turno = await orquestrador.iniciar('aprovado', '  Quero acompanhar meus pedidos.  ')

    assert turno['status'] == 'completed'
    assert turno['pergunta'] is None
    resultado = turno['resultado']
    assert resultado.status == 'aprovado_tecnicamente'
    assert resultado.descricao_cliente == 'Quero acompanhar meus pedidos.'
    assert set(resultado.pareceres_finais) == {'backend', 'frontend', 'qa'}
    assert len(resultado.rodadas) == 1
    for papel in ('backend', 'frontend', 'qa'):
        assert [schema for schema, _ in modelos[papel].chamadas] == [Parecer]


@pytest.mark.asyncio
async def test_entrevista_faz_uma_pergunta_por_vez_e_preserva_as_respostas():
    especificacao = criar_especificacao()
    perguntas = ['Quem consulta?', 'Quais pedidos?']
    modelos = criar_modelos(
        entrevistas=[
            Entrevista(especificacao=especificacao, perguntas=perguntas),
            Entrevista(especificacao=especificacao),
        ]
    )
    orquestrador = Orquestrador(modelos)

    primeiro = await orquestrador.iniciar('entrevista', 'Quero acompanhar meus pedidos.')
    assert primeiro['status'] == 'waiting_user'
    assert primeiro['pergunta'] == 'Quem consulta?'
    assert primeiro['pergunta_indice'] == 1
    assert primeiro['pergunta_total'] == len(perguntas)
    assert all(not modelos[papel].chamadas for papel in ('backend', 'frontend', 'qa'))

    segundo = await orquestrador.responder('entrevista', '  O titular.  ')
    assert segundo['status'] == 'waiting_user'
    assert segundo['pergunta'] == 'Quais pedidos?'
    assert segundo['pergunta_indice'] == len(perguntas)

    final = await orquestrador.responder('entrevista', 'Somente os próprios pedidos.')
    assert final['status'] == 'completed'
    entrevista = final['resultado'].entrevistas[0]
    assert entrevista.perguntas == ['Quem consulta?', 'Quais pedidos?']
    assert entrevista.respostas == ['O titular.', 'Somente os próprios pedidos.']
    _, mensagens = modelos['interviewer'].chamadas[1]
    assert 'Resposta do cliente: O titular.' in mensagens[1].content
    assert 'Resposta do cliente: Somente os próprios pedidos.' in mensagens[1].content


@pytest.mark.asyncio
@pytest.mark.parametrize(('texto', 'encerrar'), [(' /ENCERRAR ', False), ('', True)])
async def test_encerramento_antecipado_preserva_pergunta_sem_inventar_resposta(texto, encerrar):
    modelos = criar_modelos(
        entrevistas=[
            Entrevista(especificacao=criar_especificacao(), perguntas=['Quem consulta?']),
        ]
    )
    orquestrador = Orquestrador(modelos)
    await orquestrador.iniciar('encerramento', 'Quero acompanhar pedidos.')

    final = await orquestrador.responder('encerramento', texto, encerrar=encerrar)

    assert final['status'] == 'completed_with_pending'
    assert final['resultado'].especificacao.questoes_em_aberto == ['Quem consulta?']
    assert final['resultado'].entrevistas == []
    assert len(modelos['interviewer'].chamadas) == 1


@pytest.mark.asyncio
async def test_desacordo_encerra_no_limite_de_debates_sem_fabricar_consenso():
    especificacao = criar_especificacao()
    modelos = criar_modelos(
        consolidacoes=[
            Consolidacao(especificacao=especificacao, decisoes=[]),
        ]
    )
    modelos['backend'].padroes[Parecer] = criar_parecer(aprovado=False, gravidade='alta')
    max_debates = 1
    orquestrador = Orquestrador(modelos, config=OrquestradorConfig(max_rodadas=max_debates))

    final = await orquestrador.iniciar('desacordo', 'Quero acompanhar pedidos.')

    assert final['status'] == 'completed_with_pending'
    assert final['resultado'].pareceres_finais['backend'].aprovado is False
    assert len(final['resultado'].rodadas) == 1
    schemas_analista = [schema for schema, _ in modelos['interviewer'].chamadas]
    assert schemas_analista.count(Consolidacao) == 1
    # O limite conta debates/consolidações; há uma reavaliação final.
    schemas_backend = [schema for schema, _ in modelos['backend'].chamadas]
    assert schemas_backend.count(Parecer) == max_debates + 1


@pytest.mark.asyncio
async def test_checkpoint_sqlite_retoma_entrevista_apos_recriar_orquestrador(tmp_path):
    caminho = str(tmp_path / 'checkpoints_teste.db')
    especificacao = criar_especificacao()
    primeiros_modelos = criar_modelos(
        entrevistas=[
            Entrevista(especificacao=especificacao, perguntas=['Quem consulta?']),
        ]
    )
    async with create_checkpointer(caminho) as saver:
        primeiro = Orquestrador(primeiros_modelos, checkpointer=saver)
        turno = await primeiro.iniciar('persistido', 'Quero acompanhar pedidos.')
        assert turno['status'] == 'waiting_user'

    novos_modelos = criar_modelos()
    async with create_checkpointer(caminho) as saver:
        retomado = Orquestrador(novos_modelos, checkpointer=saver)
        final = await retomado.responder('persistido', 'Somente o titular.')

    assert final['status'] == 'completed'
    assert final['resultado'].descricao_cliente == 'Quero acompanhar pedidos.'
    assert final['resultado'].entrevistas[0].respostas == ['Somente o titular.']
    assert final['resultado'].especificacao.requisitos[0].id == 'RF-001'


@pytest.mark.asyncio
async def test_conversas_intercaladas_nao_misturam_descricao_nem_respostas():
    especificacao = criar_especificacao()
    pergunta = Entrevista(especificacao=especificacao, perguntas=['Quem consulta?'])
    resposta = Entrevista(especificacao=especificacao)
    modelos = criar_modelos(entrevistas=[pergunta, pergunta, resposta, resposta])
    orquestrador = Orquestrador(modelos)
    await orquestrador.iniciar('cliente-a', 'Projeto da loja A.')
    await orquestrador.iniciar('cliente-b', 'Projeto da loja B.')

    final_a = await orquestrador.responder('cliente-a', 'Usuários da loja A.')
    final_b = await orquestrador.responder('cliente-b', 'Usuários da loja B.')

    assert final_a['resultado'].descricao_cliente == 'Projeto da loja A.'
    assert final_b['resultado'].descricao_cliente == 'Projeto da loja B.'
    assert final_a['resultado'].entrevistas[0].respostas == ['Usuários da loja A.']
    assert final_b['resultado'].entrevistas[0].respostas == ['Usuários da loja B.']
    _, mensagens_a = modelos['interviewer'].chamadas[2]
    assert 'loja B' not in mensagens_a[1].content
    _, mensagens_b = modelos['interviewer'].chamadas[3]
    assert 'loja A' not in mensagens_b[1].content


@pytest.mark.asyncio
async def test_stream_emite_progresso_e_termina_com_resultado():
    orquestrador = Orquestrador(criar_modelos())

    eventos = [evento async for evento in orquestrador.stream('stream', 'Quero acompanhar pedidos.', iniciar=True)]

    assert eventos[-1]['tipo'] == 'done'
    assert eventos[-1]['turno']['status'] == 'completed'
    progresso = [evento['mensagem'] for evento in eventos if evento['tipo'] == 'status']
    assert 'backend concluiu sua análise.' in progresso
    assert 'frontend concluiu sua análise.' in progresso
    assert 'qa concluiu sua análise.' in progresso
    assert 'Especificação final pronta.' in progresso


@pytest.mark.asyncio
async def test_fluxo_com_um_especialista_personalizado_usa_apenas_a_equipe_escolhida():
    modelos = criar_modelos()
    especialista = EspecialistaConfig(
        chave='seguranca',
        nome='Especialista de Segurança',
        descricao='Revisa o controle de acesso.',
        modelo=modelos['backend'],
        prompt='Avalie as regras de autorização.',
    )
    orquestrador = Orquestrador({'interviewer': modelos['interviewer']}, especialistas=[especialista])

    eventos = [evento async for evento in orquestrador.stream('equipe-personalizada', 'Quero acompanhar pedidos.', iniciar=True)]

    conclusao = eventos[-1]['turno']
    assert conclusao['status'] == 'completed'
    assert set(conclusao['resultado'].pareceres_finais) == {'seguranca'}
    progresso = [evento['mensagem'] for evento in eventos if evento['tipo'] == 'status']
    assert 'Especialista de Segurança concluiu sua análise.' in progresso
    assert not modelos['frontend'].chamadas
    assert not modelos['qa'].chamadas
