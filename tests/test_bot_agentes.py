"""Testes unitários dos agentes com modelos de linguagem simulados."""

import pytest

from multi_agent_chat.chatbot.agentes.analista import Analista
from multi_agent_chat.chatbot.agentes.base import AgenteEstruturado
from multi_agent_chat.chatbot.agentes.especialistas import Especialista
from multi_agent_chat.chatbot.estados.schemas import (
    Consolidacao,
    Debate,
    Decisao,
    Entrevista,
    Parecer,
)
from tests.bot_helpers import (
    ModeloRoteirizado,
    criar_especificacao,
    criar_parecer,
)


@pytest.mark.asyncio
async def test_agente_preserva_descricao_e_contexto_no_prompt():
    esperado = criar_parecer()
    modelo = ModeloRoteirizado({Parecer: [esperado]})
    agente = AgenteEstruturado(modelo, 'Avalie sem inventar decisões.', Parecer)

    resposta = await agente.executar(
        'Avaliar acesso',
        'Pedidos com acentuação: ação e autorização.',
        {'decisao_cliente': 'Somente o titular pode consultar.'},
    )

    assert resposta == esperado
    _, mensagens = modelo.chamadas[0]
    assert mensagens[0].content == 'Avalie sem inventar decisões.'
    assert 'Avaliar acesso' in mensagens[1].content
    assert 'Pedidos com acentuação: ação e autorização.' in mensagens[1].content
    assert 'decisao_cliente: Somente o titular pode consultar.' in mensagens[1].content


@pytest.mark.asyncio
async def test_analista_recebe_especificacao_anterior_ao_reentrevistar():
    especificacao = criar_especificacao(pendencias=['Quem pode consultar?'])
    entrevista = Entrevista(especificacao=especificacao, perguntas=['Quem é o titular?'])
    modelo = ModeloRoteirizado({Entrevista: [entrevista]})
    analista = Analista(modelo)

    resposta = await analista.entrevistar('O cliente respondeu: só o titular.', especificacao)

    assert resposta == entrevista
    _, mensagens = modelo.chamadas[0]
    assert 'O cliente respondeu: só o titular.' in mensagens[1].content
    assert 'especificacao_atual:' in mensagens[1].content
    assert 'RF-001' in mensagens[1].content
    assert 'Quem pode consultar?' in mensagens[1].content


@pytest.mark.asyncio
async def test_consolidacao_preserva_pendencia_e_decisao_do_analista():
    especificacao = criar_especificacao(pendencias=['Confirmar regra de acesso.'])
    consolidacao = Consolidacao(
        especificacao=especificacao,
        decisoes=[Decisao(referencia='backend:RF-001', acao='pendente', justificativa='Depende do cliente.')],
        perguntas_cliente=['Quem pode consultar?'],
    )
    modelo = ModeloRoteirizado({Consolidacao: [consolidacao]})
    analista = Analista(modelo)

    resposta = await analista.consolidar('Relato original', especificacao, {'backend': criar_parecer(gravidade='alta')}, {})

    assert resposta.especificacao.questoes_em_aberto == ['Confirmar regra de acesso.']
    assert resposta.decisoes[0].acao == 'pendente'
    assert resposta.perguntas_cliente == ['Quem pode consultar?']
    _, mensagens = modelo.chamadas[0]
    assert 'Não está definido quem pode consultar o pedido.' in mensagens[1].content


@pytest.mark.asyncio
async def test_especialista_separa_parecer_proprio_dos_pareceres_dos_colegas():
    modelo = ModeloRoteirizado({Debate: [Debate(mensagens=[])]})
    especialista = Especialista('backend', modelo, instrucoes='Foque no controle de acesso.')
    pareceres = {
        'backend': criar_parecer(gravidade='alta'),
        'frontend': criar_parecer(),
        'qa': criar_parecer(aprovado=False),
    }

    await especialista.debater('Quero acompanhar pedidos.', criar_especificacao(), pareceres)

    _, mensagens = modelo.chamadas[0]
    assert 'Foque no controle de acesso.' in mensagens[0].content
    assert 'Não invente aprovações.' in mensagens[0].content
    contexto = mensagens[1].content
    colegas = contexto.split('pareceres_dos_colegas:', 1)[1].split('meu_parecer:', 1)[0]
    assert "'frontend':" in colegas
    assert "'qa':" in colegas
    assert "'backend':" not in colegas
    assert 'meu_parecer:' in contexto
