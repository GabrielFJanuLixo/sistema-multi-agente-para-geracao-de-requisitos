from typing import Literal

from pydantic import BaseModel, Field


class Requisito(BaseModel):
    id: str = Field(description='Identificador único, como RF-001 ou RNF-001')
    categoria: Literal['RF', 'RNF']
    descricao: str
    origem: Literal['explicito', 'inferido', 'sugerido']
    evidencia: str = Field(description='Trecho da descrição ou justificativa de inferência; sem inventar citação')
    criterios_aceitacao: list[str]


class Especificacao(BaseModel):
    requisitos: list[Requisito]
    sugestoes_nao_confirmadas: list[str]
    questoes_em_aberto: list[str]


class Entrevista(BaseModel):
    especificacao: Especificacao
    perguntas: list[str] = Field(default_factory=list, max_length=5)


class InteracaoCliente(BaseModel):
    perguntas: list[str]
    respostas: list[str]


class Observacao(BaseModel):
    requisito_id: str = Field(description='ID do requisito ou GERAL')
    gravidade: Literal['baixa', 'media', 'alta', 'critica']
    problema: str
    recomendacao: str
    destinatario: str = Field(description="Identificador do especialista destinatário ou 'analista'")


class Parecer(BaseModel):
    aprovado: bool = Field(description='Verdadeiro apenas se não houver impedimentos relevantes')
    observacoes: list[Observacao]


class Mensagem(BaseModel):
    destinatario: str = Field(description="Identificador do especialista destinatário ou 'analista'")
    requisito_id: str
    argumento: str


class Debate(BaseModel):
    mensagens: list[Mensagem]


class Decisao(BaseModel):
    referencia: str = Field(description='Ex.: backend:RF-001')
    acao: Literal['aceitar', 'rejeitar', 'pendente']
    justificativa: str


class Consolidacao(BaseModel):
    especificacao: Especificacao
    decisoes: list[Decisao]
    perguntas_cliente: list[str] = Field(default_factory=list, max_length=5)


class RegistroRodada(BaseModel):
    numero: int
    pareceres: dict[str, Parecer]
    debates: dict[str, Debate]
    decisoes: list[Decisao]


class ResultadoProjeto(BaseModel):
    descricao_cliente: str
    status: Literal['aprovado_tecnicamente', 'com_pendencias']
    especificacao: Especificacao
    rodadas: list[RegistroRodada]
    pareceres_finais: dict[str, Parecer]
    entrevistas: list[InteracaoCliente] = Field(default_factory=list)
