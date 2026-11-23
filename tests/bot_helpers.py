"""Dados de exemplo e modelo local para testar os bots sem chamar o Gemini."""

from collections import deque

from langchain_core.runnables import RunnableLambda

from multi_agent_chat.chatbot.estados.schemas import (
    Consolidacao,
    Debate,
    Entrevista,
    Especificacao,
    Observacao,
    Parecer,
    Requisito,
    ResultadoProjeto,
)


def criar_especificacao(*, pendencias=None):
    return Especificacao(
        requisitos=[
            Requisito(
                id='RF-001',
                categoria='RF',
                descricao='O cliente deve acompanhar seus pedidos.',
                origem='explicito',
                evidencia='Quero acompanhar meus pedidos.',
                criterios_aceitacao=['Um pedido existente apresenta seu status.'],
            ),
        ],
        sugestoes_nao_confirmadas=['Notificar mudanças de status por e-mail.'],
        questoes_em_aberto=list(pendencias or []),
    )


def criar_parecer(*, aprovado=True, gravidade=None):
    observacoes = []
    if gravidade is not None:
        observacoes.append(
            Observacao(
                requisito_id='RF-001',
                gravidade=gravidade,
                problema='Não está definido quem pode consultar o pedido.',
                recomendacao='Confirmar a regra de acesso com o cliente.',
                destinatario='analista',
            )
        )
    return Parecer(aprovado=aprovado, observacoes=observacoes)


def criar_pareceres():
    return {papel: criar_parecer() for papel in ('backend', 'frontend', 'qa')}


def criar_resultado(*, status='aprovado_tecnicamente', especificacao=None):
    return ResultadoProjeto(
        descricao_cliente='Quero acompanhar meus pedidos.',
        status=status,
        especificacao=especificacao or criar_especificacao(),
        rodadas=[],
        pareceres_finais=criar_pareceres(),
    )


class ModeloRoteirizado:
    """Executa o prompt real, mas devolve apenas respostas locais programadas.

    Cada schema tem uma fila. Uma chamada inesperada falha imediatamente,
    em vez de acessar a rede ou inventar uma resposta que faça o teste passar.
    """

    def __init__(self, respostas=None, *, padroes=None):
        self.respostas = {schema: deque(valores) for schema, valores in (respostas or {}).items()}
        self.padroes = padroes or {}
        self.chamadas = []

    def with_structured_output(self, schema, *, method):
        if method != 'json_schema':
            raise AssertionError(f'Método de saída inesperado: {method}')

        async def responder(prompt):
            self.chamadas.append((schema, prompt.to_messages()))
            fila = self.respostas.get(schema)
            if fila:
                resposta = fila.popleft()
            elif schema in self.padroes:
                resposta = self.padroes[schema]
            else:
                raise AssertionError(f'Resposta não programada para {schema.__name__}')
            if isinstance(resposta, Exception):
                raise resposta
            return resposta.model_copy(deep=True)

        return RunnableLambda(responder)


def criar_modelos(*, entrevistas=None, consolidacoes=None):
    especificacao = criar_especificacao()
    modelos = {
        'interviewer': ModeloRoteirizado({
            Entrevista: entrevistas or [Entrevista(especificacao=especificacao)],
            Consolidacao: consolidacoes or [],
        }),
    }
    for papel in ('backend', 'frontend', 'qa'):
        modelos[papel] = ModeloRoteirizado(
            padroes={Parecer: criar_parecer(), Debate: Debate(mensagens=[])},
        )
    return modelos
