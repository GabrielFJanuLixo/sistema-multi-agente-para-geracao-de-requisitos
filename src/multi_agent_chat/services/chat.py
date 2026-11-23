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
    EspecialistaConfig,
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
from multi_agent_chat.schemas.chat import ChatResponse, ReopenRequest
from multi_agent_chat.services.conversations import ConversationService

DEFAULT_SPECIALIST_ROLES = [
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
    if isinstance(exc, GoogleAPIError) and exc.code == HTTPStatus.SERVICE_UNAVAILABLE:
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
        'Gemini API rejected the request. Check the API key and the model settings.',
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

    @staticmethod
    def _agent_key(agent: Agent) -> str:
        if agent.role is not None:
            return agent.role.value
        return f'agent_{agent.id}'

    async def _ensure_conversation_agents(self, conversation: Conversation) -> tuple[Agent, list[Agent]]:
        agent_repo = self.conversation_service.agent_repository
        conv_repo = self.conversation_service.repository

        # 1. Garante que o entrevistador/mediador existe
        interviewer = await agent_repo.find_by_role(AgentRole.INTERVIEWER)
        if interviewer is None:
            seeded = await seed_default_agents(agent_repo.session)
            interviewer = next(a for a in seeded if a.role == AgentRole.INTERVIEWER)

        # 2. Usa os especialistas escolhidos pelo usuário na conversa
        conv_agents = await conv_repo.list_agents(conversation.id, limit=100)
        specialists = [a for a in conv_agents if a.role != AgentRole.INTERVIEWER]

        # 3. Se o usuário não escolheu nenhum, aplica os 3 default como padrão
        if not specialists:
            default_specialists = await agent_repo.list_by_roles(DEFAULT_SPECIALIST_ROLES)
            if len(default_specialists) < len(DEFAULT_SPECIALIST_ROLES):
                seeded = await seed_default_agents(agent_repo.session)
                default_specialists = [a for a in seeded if a.role in DEFAULT_SPECIALIST_ROLES]
            for agent in default_specialists:
                await conv_repo.add_agent(conversation, agent)
            specialists = default_specialists

        return interviewer, specialists

    def _build_orchestrator(
        self,
        interviewer: Agent,
        specialists: list[Agent],
        config: OrquestradorConfig,
    ) -> Orquestrador:
        especialistas_cfg = [
            EspecialistaConfig(
                chave=self._agent_key(agent),
                nome=agent.name,
                descricao=agent.description,
                modelo=self.model_factory(agent),
                prompt=agent.system_prompt,
            )
            for agent in specialists
        ]
        return Orquestrador(
            modelos={'interviewer': self.model_factory(interviewer)},
            prompts={'interviewer': interviewer.system_prompt},
            checkpointer=self.checkpointer,
            config=config,
            especialistas=especialistas_cfg,
        )

    @staticmethod
    async def _run_turn(
        orquestrador: Orquestrador,
        conversation: Conversation,
        content: str,
        *,
        finish: bool,
    ) -> dict:
        thread_id = ChatService._thread_id(conversation.user_id, conversation.id)
        try:
            if conversation.status == ConversationStatus.IDLE:
                return await orquestrador.iniciar(thread_id, content)
            return await orquestrador.responder(thread_id, content, encerrar=finish)
        except GEMINI_ERRORS as exc:
            status_code, detail = gemini_error_info(exc)
            raise HTTPException(status_code=status_code, detail=detail) from exc

    @staticmethod
    async def _run_reopen_turn(
        orquestrador: Orquestrador,
        conversation: Conversation,
        resultado_anterior: ResultadoProjeto,
        content: str,
    ) -> dict:
        thread_id = ChatService._thread_id(conversation.user_id, conversation.id)
        try:
            return await orquestrador.reabrir(
                thread_id,
                resultado_anterior,
                complemento=content,
            )
        except GEMINI_ERRORS as exc:
            status_code, detail = gemini_error_info(exc)
            raise HTTPException(status_code=status_code, detail=detail) from exc

    @staticmethod
    def _format_user_content(content: str, *, finish: bool) -> str:
        texto = content.strip()
        if texto:
            return texto
        return '/encerrar' if finish else '(sem resposta)'

    @staticmethod
    def _format_reopen_content(content: str, additional_rounds: int) -> str:
        texto = content.strip()
        if texto:
            return texto
        return f'/reabrir (+{additional_rounds} rodadas)'

    @staticmethod
    def _format_assistant_content(turno: dict) -> str:
        if turno['status'] == ConversationStatus.WAITING_USER.value:
            return formatar_pergunta_markdown(
                turno['pergunta'],
                turno['pergunta_indice'],
                turno['pergunta_total'],
            )
        return formatar_especificacao_markdown(turno['resultado'])

    async def _setup_turn(self, conversation_id: int, user_id: int, content: str):
        conversation = await self.conversation_service.get_conversation_by_id(conversation_id, user_id)
        if conversation.status in FINISHED_STATUSES:
            raise HTTPException(
                status_code=HTTPStatus.CONFLICT,
                detail='Conversation is already completed',
            )
        if conversation.status == ConversationStatus.IDLE and not (content.strip()):
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail='Initial project description cannot be empty',
            )

        interviewer, specialists = await self._ensure_conversation_agents(conversation)
        config = OrquestradorConfig(
            max_rodadas=conversation.max_rounds,
            max_entrevistas=conversation.max_interviews,
        )
        orquestrador = self._build_orchestrator(interviewer, specialists, config)
        return conversation, interviewer, orquestrador

    async def _setup_reopen(
        self,
        conversation_id: int,
        user_id: int,
        reopen_in: ReopenRequest,
    ):
        conversation = await self.conversation_service.get_conversation_by_id(conversation_id, user_id)
        if conversation.status == ConversationStatus.COMPLETED:
            raise HTTPException(
                status_code=HTTPStatus.CONFLICT,
                detail='Conversation is already completed without pending items',
            )
        if conversation.status != ConversationStatus.COMPLETED_WITH_PENDING:
            raise HTTPException(
                status_code=HTTPStatus.CONFLICT,
                detail='Only conversations completed with pending items can be reopened',
            )
        if not conversation.specification_json:
            raise HTTPException(
                status_code=HTTPStatus.CONFLICT,
                detail='Conversation has no specification to resume from',
            )

        resultado_anterior = ResultadoProjeto.model_validate_json(conversation.specification_json)
        new_max_rounds = len(resultado_anterior.rodadas) + reopen_in.additional_rounds
        new_max_interviews = len(resultado_anterior.entrevistas) + reopen_in.additional_interviews

        interviewer, specialists = await self._ensure_conversation_agents(conversation)
        config = OrquestradorConfig(
            max_rodadas=new_max_rounds,
            max_entrevistas=new_max_interviews,
        )
        orquestrador = self._build_orchestrator(interviewer, specialists, config)
        return (
            conversation,
            interviewer,
            orquestrador,
            resultado_anterior,
            (new_max_rounds, new_max_interviews),
        )

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
            update_data['specification_json'] = turno['resultado'].model_dump_json()

        await self.conversation_service.repository.update(conversation, update_data)
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

    async def _update_reopen_limits(
        self,
        conversation: Conversation,
        max_rounds: int,
        max_interviews: int,
    ) -> None:
        await self.conversation_service.repository.update(
            conversation,
            {
                'max_rounds': max_rounds,
                'max_interviews': max_interviews,
            },
        )

    @staticmethod
    async def _stream_events(source, on_done):
        yield format_sse_event(
            data_str=json.dumps({'mensagem': 'Processando...'}),
            event='status',
        )
        turno = None
        try:
            async for item in source:
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

        result = await on_done(turno)

        done_json = ChatResponse.model_validate(result, from_attributes=True).model_dump_json()

        yield format_sse_event(data_str=done_json, event='done')

    async def send_message(
        self,
        conversation_id: int,
        user_id: int,
        content: str,
        *,
        finish: bool = False,
    ) -> dict:
        conversation, interviewer, orquestrador = await self._setup_turn(conversation_id, user_id, content)
        turno = await self._run_turn(orquestrador, conversation, content, finish=finish)
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
        conversation, interviewer, orquestrador = await self._setup_turn(conversation_id, user_id, content)
        thread_id = self._thread_id(user_id, conversation_id)
        is_start = conversation.status == ConversationStatus.IDLE
        user_content = self._format_user_content(content, finish=finish)
        source = orquestrador.stream(thread_id, content, iniciar=is_start, encerrar=finish)

        async def on_done(turno):
            return await self._save_turn(conversation, interviewer, user_content, turno)

        return self._stream_events(source, on_done)

    async def reopen_conversation(
        self,
        conversation_id: int,
        user_id: int,
        reopen_in: ReopenRequest,
    ) -> dict:
        (
            conversation,
            interviewer,
            orquestrador,
            resultado_anterior,
            (new_max_rounds, new_max_interviews),
        ) = await self._setup_reopen(conversation_id, user_id, reopen_in)
        turno = await self._run_reopen_turn(
            orquestrador,
            conversation,
            resultado_anterior,
            reopen_in.content,
        )
        result = await self._save_turn(
            conversation,
            interviewer,
            self._format_reopen_content(reopen_in.content, reopen_in.additional_rounds),
            turno,
        )
        await self._update_reopen_limits(
            conversation,
            new_max_rounds,
            new_max_interviews,
        )
        return result

    async def stream_reopen(
        self,
        conversation_id: int,
        user_id: int,
        reopen_in: ReopenRequest,
    ):
        (
            conversation,
            interviewer,
            orquestrador,
            resultado_anterior,
            (new_max_rounds, new_max_interviews),
        ) = await self._setup_reopen(conversation_id, user_id, reopen_in)
        thread_id = self._thread_id(user_id, conversation_id)
        user_content = self._format_reopen_content(reopen_in.content, reopen_in.additional_rounds)
        source = orquestrador.stream_reabertura(
            thread_id,
            resultado_anterior,
            complemento=reopen_in.content,
        )

        async def on_done(turno):
            result = await self._save_turn(conversation, interviewer, user_content, turno)
            await self._update_reopen_limits(
                conversation,
                new_max_rounds,
                new_max_interviews,
            )
            return result

        return self._stream_events(source, on_done)

    async def get_specification(self, conversation_id: int, user_id: int) -> dict:
        conversation = await self.conversation_service.get_conversation_by_id(conversation_id, user_id)
        if not conversation.specification_json:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail='Specification not generated yet',
            )

        resultado = ResultadoProjeto.model_validate_json(conversation.specification_json)
        return {
            'conversation_id': conversation_id,
            'status': conversation.status,
            'specification': resultado,
            'markdown': formatar_especificacao_markdown(resultado),
        }
