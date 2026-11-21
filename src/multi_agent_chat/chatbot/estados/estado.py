from operator import or_
from typing import Annotated, NotRequired, TypedDict

from multi_agent_chat.chatbot.estados.schemas import (
    Debate,
    Especificacao,
    InteracaoCliente,
    Parecer,
    RegistroRodada,
    ResultadoProjeto,
)


class EstadoProjeto(TypedDict):
    descricao: str
    especificacao: Especificacao
    numero_rodada: int
    pareceres: Annotated[dict[str, Parecer], or_]
    debates: Annotated[dict[str, Debate], or_]
    consenso: bool
    rodadas: list[RegistroRodada]
    entrevistas: list[InteracaoCliente]
    perguntas: list[str]
    respostas_atuais: list[str]
    cliente_encerrado: bool
    fase: str
    acao: str
    resultado: NotRequired[ResultadoProjeto]
