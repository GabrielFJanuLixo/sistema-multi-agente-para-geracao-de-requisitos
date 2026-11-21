from multi_agent_chat.chatbot.agentes.base import AgenteEstruturado
from multi_agent_chat.chatbot.estados.schemas import Consolidacao, Entrevista

INSTRUCOES = """
Você é o Entrevistador/Avaliador e analista sênior de requisitos de software.
Sua fonte de verdade sobre o negócio é SOMENTE o relato e as respostas do cliente.
Separe explicitamente requisitos pedidos, inferidos e sugestões.
Não trate sugestões de especialistas como decisões confirmadas do cliente.
Descreva comportamento observável, sem impor framework ou tecnologia sem necessidade.
Requisitos devem ser claros, sem duplicação, e ter critérios de aceitação testáveis.
A evidência deve ser um trecho real do relato ou uma justificativa honesta; não invente citações.
Faça perguntas claras ao cliente quando faltar uma decisão de negócio essencial.
Não pergunte sobre decisões puramente técnicas que os especialistas podem resolver.
Não repita perguntas já respondidas. Se o cliente não souber, deixe a questão em aberto.
Nunca invente uma resposta do cliente para eliminar uma pendência.
Preserve IDs estáveis e os requisitos já confirmados ao incorporar novas informações.
"""

PROMPT_ENTREVISTA = """
Prepare ou atualize a especificação com base na conversa completa.
Retorne em perguntas apenas dúvidas essenciais ainda não respondidas, no máximo cinco por vez.
Retorne perguntas vazias quando for possível seguir para a avaliação técnica.
Mantenha em questoes_em_aberto as decisões que o cliente ainda não conseguiu esclarecer.
"""

PROMPT = """
Você é também o mediador. Para cada observação relevante, registre decisão:
aceitar, rejeitar ou pendente, sempre com justificativa.
As mensagens entre especialistas são pareceres, não fatos confirmados.
Resolva as divergências técnicas. Quando a solução depender do cliente, formule
perguntas_cliente objetivas, no máximo cinco, e mantenha essas dúvidas na especificação.
Não repita perguntas que o cliente já respondeu ou disse não saber.
"""


class Analista:
    def __init__(self, modelo, instrucoes: str | None = None):
        base = instrucoes or INSTRUCOES

        # Define a função de entrevistador e consolidador
        self.entrevistador = AgenteEstruturado(
            modelo, base + PROMPT_ENTREVISTA, Entrevista
        )
        self.consolidador = AgenteEstruturado(
            modelo, base + PROMPT, Consolidacao
        )

    # Executa a entrevista com o cliente, atualizando a especificação e gerando perguntas
    async def entrevistar(self, descricao, especificacao=None):
        contexto = (
            {'especificacao_atual': especificacao}
            if especificacao is not None
            else {}
        )
        return await self.entrevistador.executar(
            'Entenda o pedido, atualize os requisitos e pergunte o que '
            'precisa esclarecer.',
            descricao,
            contexto,
        )

    # Consolida as respostas do cliente, resolve divergências técnicas e atualiza a especificação
    async def consolidar(self, descricao, especificacao, pareceres, debates):
        return await self.consolidador.executar(
            'Resolva as críticas técnicas e consolide nova versão. '
            'Registre decisões e perguntas ao cliente quando faltar informação de negócio.',
            descricao,
            {
                'especificacao_atual': especificacao,
                'pareceres': pareceres,
                'debates': debates,
            },
        )
