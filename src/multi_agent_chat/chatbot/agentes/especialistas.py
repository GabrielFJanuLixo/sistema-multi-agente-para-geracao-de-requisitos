from multi_agent_chat.chatbot.agentes.base import AgenteEstruturado
from multi_agent_chat.chatbot.estados.schemas import Debate, Parecer

PAPEIS = {
    'backend': """Você é um especialista de backend e dados.
Analise persistência, operações, regras de negócio, segurança e integração.
Não imponha tecnologias específicas; identifique requisitos e ambiguidades.
""",
    'frontend': """Você é um especialista em frontend e UX.
Analise jornada, navegação, acessibilidade, estados de interface e erros.
Não invente funcionalidades que o cliente não confirmou.
""",
    'qa': """Você é um engenheiro QA independente e rigoroso.
Identifique ambiguidades, exceções, conflitos e falta de critérios testáveis.
Não se limite à correção gramatical; considere requisitos ausentes e contraditórios.
""",
}

RESTRICAO = """
A descrição do cliente é a única evidência de negócio.
Não invente aprovações. Marque como crítico só bloqueios reais.
Direcione problemas à especialidade apropriada ou ao analista.
"""

PROMPT_AVALIACAO = """
Avalie os requisitos de forma independente.
"""

PROMPT_DEBATE_TEMPLATE = """
Responda criticamente às observações dos colegas com argumentos úteis.
Enderece mensagens somente a um destes identificadores: {destinatarios}.
Equipe participante:
{equipe}
Discorde quando necessário, sem buscar consenso artificial.
Se não houver nada relevante a discutir, retorne mensagens vazias.
"""


class Especialista:
    def __init__(
        self,
        nome,
        modelo,
        instrucoes: str | None = None,
        equipe: dict[str, str] | None = None,
    ):
        # Define o papel do especialista, criando avaliador e debatedor
        self.nome = nome
        prompt_base = instrucoes or PAPEIS.get(nome)
        if not prompt_base:
            raise ValueError(f'Instruções obrigatórias para o especialista {nome!r}')
        equipe_map = equipe or {k: k for k in PAPEIS}
        destinatarios = ', '.join([*equipe_map.keys(), 'analista'])
        equipe_texto = '\n'.join(f'- {chave}: {desc}' for chave, desc in equipe_map.items())
        prompt_debate = PROMPT_DEBATE_TEMPLATE.format(
            destinatarios=destinatarios,
            equipe=equipe_texto,
        )
        base = prompt_base + RESTRICAO
        self.avaliador = AgenteEstruturado(
            modelo,
            base + PROMPT_AVALIACAO,
            Parecer,
        )
        self.debatedor = AgenteEstruturado(
            modelo,
            base + prompt_debate,
            Debate,
        )

    # Avalia os requisitos e produz um parecer, considerando mensagens de colegas quando existirem
    async def avaliar(self, descricao, especificacao, mensagens_recebidas=None):
        return await self.avaliador.executar(
            'Avalie requisitos e produza um parecer: aprovado e observacoes. Considere mensagens dos colegas quando existirem.',
            descricao,
            {
                'especificacao': especificacao,
                'mensagens_dos_colegas': mensagens_recebidas or [],
            },
        )

    # Debata com os demais especialistas, produzindo
    # mensagens endereçadas aos colegas quando houver discordância,
    # dependência ou esclarecimento.
    async def debater(self, descricao, especificacao, pareceres):
        return await self.debatedor.executar(
            'Debata com os demais especialistas. Produza mensagens endereçadas '
            'aos colegas quando houver discordância, dependência ou esclarecimento.',
            descricao,
            {
                'especificacao': especificacao,
                'pareceres_dos_colegas': {nome: parecer for nome, parecer in pareceres.items() if nome != self.nome},
                'meu_parecer': pareceres[self.nome],
            },
        )
