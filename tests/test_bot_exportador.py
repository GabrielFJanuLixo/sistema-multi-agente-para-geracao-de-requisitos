"""Verifica as informações que devem aparecer na especificação exportada."""

from multi_agent_chat.chatbot.controle.exportador import (
    formatar_especificacao_markdown,
    formatar_pergunta_markdown,
)
from multi_agent_chat.chatbot.estados.schemas import InteracaoCliente
from tests.bot_helpers import criar_especificacao, criar_parecer, criar_resultado


def test_exportacao_preserva_requisito_e_distingue_sugestoes_de_pendencias():
    especificacao = criar_especificacao(pendencias=['Quem pode consultar o pedido?'])
    resultado = criar_resultado(status='com_pendencias', especificacao=especificacao)

    markdown = formatar_especificacao_markdown(resultado)

    assert '**Status da revisão:** com_pendencias' in markdown
    assert '### RF-001 — RF (explicito)' in markdown
    assert 'O cliente deve acompanhar seus pedidos.' in markdown
    assert '**Evidência/justificativa:** Quero acompanhar meus pedidos.' in markdown
    assert '- Um pedido existente apresenta seu status.' in markdown
    sugestoes = markdown.split('## Sugestões não confirmadas\n', 1)[1].split('## Questões em aberto', 1)[0]
    assert '- Notificar mudanças de status por e-mail.' in sugestoes
    pendencias = markdown.split('## Questões em aberto\n', 1)[1].split('## Pendências técnicas', 1)[0]
    assert '- Quem pode consultar o pedido?' in pendencias


def test_exportacao_inclui_criticas_e_entrevista_sem_inventar_resposta():
    resultado = criar_resultado(status='com_pendencias')
    resultado.pareceres_finais['qa'] = criar_parecer(aprovado=False, gravidade='critica')
    resultado.entrevistas = [
        InteracaoCliente(
            perguntas=['Quem consulta?', 'Qual o prazo?'],
            respostas=['Somente o titular.', ''],
        )
    ]

    markdown = formatar_especificacao_markdown(resultado)

    assert '**qa / RF-001 (critica)**' in markdown
    assert 'Não está definido quem pode consultar o pedido.' in markdown
    assert 'Confirmar a regra de acesso com o cliente.' in markdown
    assert '**Pergunta:** Quem consulta?\n**Resposta:** Somente o titular.' in markdown
    assert '**Pergunta:** Qual o prazo?\n**Resposta:** Não respondida.' in markdown
    assert resultado.entrevistas[0].respostas == ['Somente o titular.', '']


def test_pergunta_exportada_mostra_posicao_e_comando_de_encerramento():
    markdown = formatar_pergunta_markdown('Quem consulta?', 2, 3)

    assert '**Pergunta 2 de 3:**' in markdown
    assert 'Quem consulta?' in markdown
    assert '`/encerrar`' in markdown
