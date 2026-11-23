import json
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import Depends, HTTPException
from fastapi.sse import format_sse_event
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleGenerativeAIError,
    GoogleRateLimitError,
)

from multi_agent_chat.chatbot.controle.exportador import (
    formatar_especificacao_markdown,
    formatar_pergunta_markdown,
)
from multi_agent_chat.chatbot.controle.orquestrador import (
    Orquestrador,
    OrquestradorConfig,
)
from multi_agent_chat.chatbot.estados.schemas import ResultadoProjeto
from multi_agent_chat.core.chatbot import (
    ModelFactory,
    get_checkpointer,
    get_model_factory,
)
from multi_agent_chat.core.seed import seed_default_agents
from multi_agent_chat.models.agent import Agent, AgentRole
from multi_agent_chat.models.conversation import (
    Conversation,
    ConversationStatus,
)
from multi_agent_chat.repositories.messages import MessageRepository
from multi_agent_chat.schemas.chat import ChatResponse
from multi_agent_chat.services.conversations import ConversationService

REQUIRED_ROLES = [
    AgentRole.INTERVIEWER,
    AgentRole.BACKEND,
    AgentRole.FRONTEND,
    AgentRole.QA,
]
FINISHED_STATUSES = {
    ConversationStatus.COMPLETED,
    ConversationStatus.COMPLETED_WITH_PENDING,
}

# Qualquer falha vinda do Gemini: erros 5xx (GoogleAPIError) e os demais
# (GoogleGenerativeAIError: requisição inválida, chave inválida, modelo não
# encontrado, permissão negada, limite de cota...).
GEMINI_ERRORS = (GoogleAPIError, GoogleGenerativeAIError)


def gemini_error_info(exc: Exception) -> tuple[HTTPStatus, str]:
    """Traduz uma falha do Gemini em (status HTTP, mensagem segura).

    A mensagem não repassa o texto cru da exceção, que pode trazer detalhes
    internos da chamada.
    """
    if isinstance(exc, GoogleRateLimitError):
        return (
            HTTPStatus.TOO_MANY_REQUESTS,
            'Gemini API rate limit exceeded. Try again later.',
        )
    if (
        isinstance(exc, GoogleAPIError)
        and exc.code == HTTPStatus.SERVICE_UNAVAILABLE
    ):
        return (
            HTTPStatus.SERVICE_UNAVAILABLE,
            'Gemini API is temporarily unavailable.',
        )
    if isinstance(exc, GoogleAPIError):
        return (
            HTTPStatus.BAD_GATEWAY,
            'Error communicating with Gemini API.',
        )
    return (
        HTTPStatus.BAD_GATEWAY,
        'Gemini API rejected the request. '
        'Check the API key and the model settings.',
    )


ConversationServiceDep = Annotated[ConversationService, Depends()]
MessageRepositoryDep = Annotated[MessageRepository, Depends()]
ModelFactoryDep = Annotated[ModelFactory, Depends(get_model_factory)]
CheckpointerDep = Annotated[Any, Depends(get_checkpointer)]


class ChatService:
    def __init__(
        self,
        conversation_service: ConversationServiceDep,
        message_repository: MessageRepositoryDep,
        model_factory: ModelFactoryDep,
        checkpointer: CheckpointerDep,
    ):
        self.conversation_service = conversation_service
        self.message_repository = message_repository
        self.model_factory = model_factory
        self.checkpointer = checkpointer

    @staticmethod
    def _thread_id(user_id: int, conversation_id: int) -> str:
        return f'user:{user_id}:conversation:{conversation_id}'

    async def _ensure_conversation_agents(
        self, conversation: Conversation
    ) -> dict[AgentRole, Agent]:
        agent_repo = self.conversation_service.agent_repository
        conv_repo = self.conversation_service.repository

        agents = await agent_repo.list_by_roles(REQUIRED_ROLES)
        if len(agents) < len(REQUIRED_ROLES):
            agents = await seed_default_agents(agent_repo.session)

        agents_by_role = {
            agent.role: agent for agent in agents if agent.role is not None
        }
        for role in REQUIRED_ROLES:
            await conv_repo.add_agent(conversation, agents_by_role[role])

        return agents_by_role

    def _build_orchestrator(
        self,
        conversation: Conversation,
        agents_by_role: dict[AgentRole, Agent],
    ) -> Orquestrador:
        modelos = {
            role.value: self.model_factory(agent)
            for role, agent in agents_by_role.items()
        }
        prompts = {
            role.value: agent.system_prompt
            for role, agent in agents_by_role.items()
        }
        config = OrquestradorConfig(
            max_rodadas=conversation.max_rounds,
            max_entrevistas=conversation.max_interviews,
        )
        return Orquestrador(
            modelos,
            prompts=prompts,
            checkpointer=self.checkpointer,
            config=config,
        )

    @staticmethod
    async def _run_turn(
        orquestrador: Orquestrador,
        conversation: Conversation,
        content: str,
        *,
        finish: bool,
    ) -> dict:
        thread_id = ChatService._thread_id(
            conversation.user_id, conversation.id
        )
        try:
            if conversation.status == ConversationStatus.IDLE:
                return await orquestrador.iniciar(thread_id, content)
            return await orquestrador.responder(
                thread_id, content, encerrar=finish
            )
        except GEMINI_ERRORS as exc:
            status_code, detail = gemini_error_info(exc)
            raise HTTPException(
                status_code=status_code, detail=detail
            ) from exc

    @staticmethod
    def _format_user_content(content: str, *, finish: bool) -> str:
        texto = content.strip()
        if texto:
            return texto
        return '/encerrar' if finish else '(sem resposta)'

    @staticmethod
    def _format_assistant_content(turno: dict) -> str:
        if turno['status'] == ConversationStatus.WAITING_USER.value:
            return formatar_pergunta_markdown(
                turno['pergunta'],
                turno['pergunta_indice'],
                turno['pergunta_total'],
            )
        return formatar_especificacao_markdown(turno['resultado'])

    async def _setup_turn(
        self, conversation_id: int, user_id: int, content: str
    ):
        conversation = (
            await self.conversation_service.get_conversation_by_id(
                conversation_id, user_id
            )
        )
        if conversation.status in FINISHED_STATUSES:
            raise HTTPException(
                status_code=HTTPStatus.CONFLICT,
                detail='Conversation is already completed',
            )
        if conversation.status == ConversationStatus.IDLE and not (
            content.strip()
        ):
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail='Initial project description cannot be empty',
            )

        agents_by_role = await self._ensure_conversation_agents(conversation)
        orquestrador = self._build_orchestrator(conversation, agents_by_role)
        return conversation, agents_by_role[AgentRole.INTERVIEWER], orquestrador

    async def _save_turn(
        self,
        conversation: Conversation,
        interviewer: Agent,
        user_content: str,
        turno: dict,
    ) -> dict:
        user_message = await self.message_repository.create({
            'conversation_id': conversation.id,
            'user_id': conversation.user_id,
            'content': user_content,
        })
        assistant_message = await self.message_repository.create({
            'conversation_id': conversation.id,
            'agent_id': interviewer.id,
            'content': self._format_assistant_content(turno),
        })

        new_status = ConversationStatus(turno['status'])
        update_data: dict[str, Any] = {'status': new_status}
        if turno['resultado'] is not None:
            update_data['specification_json'] = turno[
                'resultado'
            ].model_dump_json()

        await self.conversation_service.repository.update(
            conversation, update_data
        )
        return {
            'conversation_id': conversation.id,
            'status': new_status,
            'user_message': user_message,
            'assistant_message': assistant_message,
            'question': turno['pergunta'],
            'question_index': turno['pergunta_indice'],
            'question_total': turno['pergunta_total'],
            'specification': turno['resultado'],
        }

    async def send_message(
        self,
        conversation_id: int,
        user_id: int,
        content: str,
        *,
        finish: bool = False,
    ) -> dict:
        conversation, interviewer, orquestrador = await self._setup_turn(
            conversation_id, user_id, content
        )
        turno = await self._run_turn(
            orquestrador, conversation, content, finish=finish
        )
        return await self._save_turn(
            conversation,
            interviewer,
            self._format_user_content(content, finish=finish),
            turno,
        )

    async def stream_message(
        self,
        conversation_id: int,
        user_id: int,
        content: str,
        *,
        finish: bool = False,
    ):
        conversation, interviewer, orquestrador = await self._setup_turn(
            conversation_id, user_id, content
        )
        thread_id = self._thread_id(user_id, conversation_id)
        is_start = conversation.status == ConversationStatus.IDLE
        user_content = self._format_user_content(content, finish=finish)

        async def events():
            yield format_sse_event(
                data_str=json.dumps({'mensagem': 'Processando...'}),
                event='status',
            )
            turno = None
            try:
                async for item in orquestrador.stream(
                    thread_id, content, iniciar=is_start, encerrar=finish
                ):
                    if item['tipo'] == 'status':
                        yield format_sse_event(
                            data_str=json.dumps(
                                {'mensagem': item['mensagem']},
                                ensure_ascii=False,
                            ),
                            event='status',
                        )
                    else:
                        turno = item['turno']
            except GEMINI_ERRORS as exc:
                status_code, detail = gemini_error_info(exc)
                yield format_sse_event(
                    data_str=json.dumps({
                        'detail': detail,
                        'status_code': int(status_code),
                    }),
                    event='error',
                )
                return

            result = await self._save_turn(
                conversation, interviewer, user_content, turno
            )

            done_json = ChatResponse.model_validate(
                result, from_attributes=True
            ).model_dump_json()

            yield format_sse_event(data_str=done_json, event='done')

        return events()

    async def get_specification(
        self, conversation_id: int, user_id: int
    ) -> dict:
        conversation = (
            await self.conversation_service.get_conversation_by_id(
                conversation_id, user_id
            )
        )
        if not conversation.specification_json:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail='Specification not generated yet',
            )

        resultado = ResultadoProjeto.model_validate_json(
            conversation.specification_json
        )
        return {
            'conversation_id': conversation_id,
            'status': conversation.status,
            'specification': resultado,
            'markdown': formatar_especificacao_markdown(resultado),
        }
