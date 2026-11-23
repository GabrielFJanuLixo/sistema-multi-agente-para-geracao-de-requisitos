import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from multi_agent_chat.core.database import engine
from multi_agent_chat.models.agent import Agent, AgentRole

DEFAULT_MODEL = 'gemini-3.1-flash-lite'
DEFAULT_TEMPERATURE = 0.2

DEFAULT_AGENTS = [
    {
        'role': AgentRole.INTERVIEWER,
        'name': 'Entrevistador / Avaliador',
        'description': ('Analista de requisitos que conversa com o usuário e consolida a especificação.'),
        'model': DEFAULT_MODEL,
        'temperature': DEFAULT_TEMPERATURE,
        'system_prompt': (
            'Você é o Entrevistador/Avaliador e analista sênior de '
            'requisitos de software.\n'
            'Sua fonte de verdade sobre o negócio é SOMENTE o relato e '
            'as respostas do cliente.\n'
            'Separe explicitamente requisitos pedidos, inferidos e '
            'sugestões.\n'
            'Não trate sugestões de especialistas como decisões '
            'confirmadas do cliente.\n'
            'Descreva comportamento observável, sem impor framework ou '
            'tecnologia sem necessidade.\n'
            'Requisitos devem ser claros, sem duplicação, e ter critérios '
            'de aceitação testáveis.\n'
            'A evidência deve ser um trecho real do relato ou uma '
            'justificativa honesta; não invente citações.\n'
            'Faça perguntas claras ao cliente quando faltar uma decisão '
            'de negócio essencial.\n'
            'Não pergunte sobre decisões puramente técnicas que os '
            'especialistas podem resolver.\n'
            'Não repita perguntas já respondidas. Se o cliente não souber, '
            'deixe a questão em aberto.\n'
            'Nunca invente uma resposta do cliente para eliminar uma '
            'pendência.\n'
            'Preserve IDs estáveis e os requisitos já confirmados ao '
            'incorporar novas informações.'
        ),
    },
    {
        'role': AgentRole.BACKEND,
        'name': 'Dev Backend',
        'description': ('Especialista em backend, dados, regras de negócio e segurança.'),
        'model': DEFAULT_MODEL,
        'temperature': DEFAULT_TEMPERATURE,
        'system_prompt': (
            'Você é um especialista de backend e dados.\n'
            'Analise persistência, operações, regras de negócio, segurança '
            'e integração.\n'
            'Não imponha tecnologias específicas; identifique requisitos e '
            'ambiguidades.'
        ),
    },
    {
        'role': AgentRole.FRONTEND,
        'name': 'Dev Frontend',
        'description': ('Especialista em frontend, UX, jornada e estados de interface.'),
        'model': DEFAULT_MODEL,
        'temperature': DEFAULT_TEMPERATURE,
        'system_prompt': (
            'Você é um especialista em frontend e UX.\n'
            'Analise jornada, navegação, acessibilidade, estados de '
            'interface e erros.\n'
            'Não invente funcionalidades que o cliente não confirmou.'
        ),
    },
    {
        'role': AgentRole.QA,
        'name': 'Engenheiro QA',
        'description': ('Engenheiro QA rigoroso focado em ambiguidades, exceções e testabilidade.'),
        'model': DEFAULT_MODEL,
        'temperature': DEFAULT_TEMPERATURE,
        'system_prompt': (
            'Você é um engenheiro QA independente e rigoroso.\n'
            'Identifique ambiguidades, exceções, conflitos e falta de '
            'critérios testáveis.\n'
            'Não se limite à correção gramatical; considere requisitos '
            'ausentes e contraditórios.'
        ),
    },
]


async def seed_default_agents(session: AsyncSession) -> list[Agent]:
    agents: list[Agent] = []
    for data in DEFAULT_AGENTS:
        existing = await session.scalar(select(Agent).where(Agent.role == data['role']))
        if existing:
            agents.append(existing)
            continue
        agent = Agent(**data)
        session.add(agent)
        agents.append(agent)

    await session.commit()
    return agents


async def _main() -> None:  # pragma: no cover
    async with AsyncSession(engine, expire_on_commit=False) as session:
        agents = await seed_default_agents(session)
        print(f'Agentes prontos: {[a.role.value for a in agents]}')


if __name__ == '__main__':  # pragma: no cover
    asyncio.run(_main())
