"""Regras de aprovação, pendências e encaminhamento entre os agentes."""

from unittest.mock import AsyncMock

import pytest

from multi_agent_chat.chatbot.controle.orquestrador import EspecialistaConfig, Orquestrador, OrquestradorConfig
from multi_agent_chat.chatbot.estados.schemas import Debate, Mensagem, RegistroRodada
from tests.bot_helpers import (
    criar_especificacao,
    criar_modelos,
    criar_parecer,
    criar_pareceres,
)


@pytest.fixture
def orquestrador():
    return Orquestrador(criar_modelos())


@pytest.mark.parametrize('gravidade', [None, 'baixa', 'media'])
def test_aprovacao_exige_tres_pareceres_favoraveis_sem_bloqueios(orquestrador, gravidade):
    pareceres = criar_pareceres()
    pareceres['backend'] = criar_parecer(gravidade=gravidade)

    assert orquestrador.aprovado_tecnicamente(pareceres) is True


@pytest.mark.parametrize('papel', ['backend', 'frontend', 'qa'])
@pytest.mark.parametrize('problema', ['ausente', 'reprovado', 'alta', 'critica'])
def test_aprovacao_rejeita_parecer_ausente_reprovado_ou_bloqueante(orquestrador, papel, problema):
    pareceres = criar_pareceres()
    if problema == 'ausente':
        del pareceres[papel]
    elif problema == 'reprovado':
        pareceres[papel] = criar_parecer(aprovado=False)
    else:
        pareceres[papel] = criar_parecer(gravidade=problema)

    assert orquestrador.aprovado_tecnicamente(pareceres) is False


def test_nenhum_parecer_nao_significa_aprovacao(orquestrador):
    assert orquestrador.aprovado_tecnicamente({}) is False


@pytest.mark.parametrize(
    ('situacao', 'esperado'),
    [('aprovado', True), ('ausente', False), ('reprovado', False), ('critica', False), ('parecer-extra', False)],
)
def test_aprovacao_considera_a_equipe_personalizada_em_vez_dos_papeis_padrao(situacao, esperado):
    modelos = criar_modelos()
    especialista = EspecialistaConfig(
        chave='seguranca',
        nome='Especialista de Segurança',
        descricao='Revisa o controle de acesso.',
        modelo=modelos['backend'],
        prompt='Avalie as regras de autorização.',
    )
    orquestrador = Orquestrador({'interviewer': modelos['interviewer']}, especialistas=[especialista])
    pareceres = {'seguranca': criar_parecer()}
    if situacao == 'ausente':
        pareceres = {}
    elif situacao == 'reprovado':
        pareceres['seguranca'] = criar_parecer(aprovado=False)
    elif situacao == 'critica':
        pareceres['seguranca'] = criar_parecer(gravidade='critica')
    elif situacao == 'parecer-extra':
        pareceres['backend'] = criar_parecer()

    assert orquestrador.aprovado_tecnicamente(pareceres) is esperado


@pytest.mark.parametrize(
    'cenario',
    [
        (True, [], [], False, 'aprovado_tecnicamente'),
        (False, [], [], False, 'com_pendencias'),
        (True, ['Confirmar acesso.'], [], False, 'com_pendencias'),
        (True, [], ['Quem pode consultar?'], False, 'com_pendencias'),
        (True, [], [], True, 'com_pendencias'),
    ],
    ids=['aprovado', 'sem-consenso', 'pendencia-especificacao', 'pergunta-cliente', 'encerramento-antecipado'],
)
def test_resultado_final_so_aprova_sem_pendencias_e_sem_encerramento_antecipado(orquestrador, cenario):
    consenso, pendencias, perguntas, cliente_encerrado, status = cenario
    estado = {
        'descricao': 'Quero acompanhar pedidos.',
        'especificacao': criar_especificacao(pendencias=pendencias),
        'rodadas': [],
        'numero_rodada': 1,
        'pareceres': criar_pareceres(),
        'perguntas': perguntas,
        'consenso': consenso,
        'cliente_encerrado': cliente_encerrado,
        'entrevistas': [],
    }

    resultado = orquestrador._finalizar(estado)['resultado']

    assert resultado.status == status
    assert resultado.especificacao.questoes_em_aberto == pendencias + perguntas


def test_finalizacao_preserva_historico_e_nao_duplica_pendencias(orquestrador):
    especificacao = criar_especificacao(pendencias=['Quem pode consultar?'])
    rodada = RegistroRodada(numero=1, pareceres=criar_pareceres(), debates={}, decisoes=[])
    estado = {
        'descricao': 'Quero acompanhar pedidos.',
        'especificacao': especificacao,
        'rodadas': [rodada],
        'numero_rodada': 2,
        'pareceres': criar_pareceres(),
        'perguntas': ['Quem pode consultar?', 'Qual o prazo?'],
        'consenso': True,
        'cliente_encerrado': False,
        'entrevistas': [],
    }

    resultado = orquestrador._finalizar(estado)['resultado']

    assert resultado.especificacao.questoes_em_aberto == ['Quem pode consultar?', 'Qual o prazo?']
    assert [item.numero for item in resultado.rodadas] == [1, 2]
    assert resultado.rodadas[0] == rodada
    assert especificacao.questoes_em_aberto == ['Quem pode consultar?']
    assert estado['rodadas'] == [rodada]


@pytest.mark.parametrize(
    ('encerrado', 'entrevistas', 'destino'),
    [(False, [], 'cliente'), (True, [], ['backend', 'frontend', 'qa']), (False, [1, 2], ['backend', 'frontend', 'qa'])],
)
def test_perguntas_respeitam_encerramento_e_limite_de_entrevistas(encerrado, entrevistas, destino):
    orquestrador = Orquestrador(criar_modelos(), config=OrquestradorConfig(max_entrevistas=2))
    estado = {
        'acao': 'consultar',
        'perguntas': ['Quem pode consultar?'],
        'cliente_encerrado': encerrado,
        'entrevistas': entrevistas,
    }

    assert orquestrador._proximo(estado) == destino


@pytest.mark.asyncio
async def test_mensagens_tecnicas_chegam_somente_ao_destinatario(orquestrador):
    recebido = criar_parecer()
    backend = orquestrador.especialistas['backend']
    backend.avaliar = AsyncMock(return_value=recebido)
    estado = {
        'descricao': 'Quero acompanhar pedidos.',
        'acao': 'avaliar',
        'especificacao': criar_especificacao(),
        'entrevistas': [],
        'debates': {
            'qa': Debate(
                mensagens=[
                    Mensagem(destinatario='backend', requisito_id='RF-001', argumento='Definir autorização.'),
                    Mensagem(destinatario='frontend', requisito_id='RF-001', argumento='Definir tela.'),
                ]
            ),
        },
    }

    retorno = await orquestrador._especialista('backend', estado)

    assert retorno == {'pareceres': {'backend': recebido}}
    mensagens = backend.avaliar.await_args.args[2]
    assert mensagens == [
        {
            'remetente': 'qa',
            'destinatario': 'backend',
            'requisito_id': 'RF-001',
            'argumento': 'Definir autorização.',
        }
    ]
