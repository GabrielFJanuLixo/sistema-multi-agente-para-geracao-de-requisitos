from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial

import aiosqlite
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Overwrite, interrupt

from multi_agent_chat.chatbot.agentes.analista import Analista
from multi_agent_chat.chatbot.agentes.especialistas import PAPEIS, Especialista
from multi_agent_chat.chatbot.estados.estado import EstadoProjeto
from multi_agent_chat.chatbot.estados.schemas import (
    InteracaoCliente,
    RegistroRodada,
    ResultadoProjeto,
)

SCHEMA_MODULE = 'multi_agent_chat.chatbot.estados.schemas'
SCHEMA_TYPES = (
    'Requisito',
    'Especificacao',
    'Entrevista',
    'InteracaoCliente',
    'Observacao',
    'Parecer',
    'Mensagem',
    'Debate',
    'Decisao',
    'Consolidacao',
    'RegistroRodada',
    'ResultadoProjeto',
)

STATUS_BOTS = {
    'entrevistador_avaliador': 'Entrevistador analisando os requisitos...',
    'backend': 'Especialista de backend concluiu sua análise.',
    'frontend': 'Especialista de frontend concluiu sua análise.',
    'qa': 'Especialista de QA concluiu sua análise.',
    'resposta': 'Especificação final pronta.',
}


def build_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            (SCHEMA_MODULE, nome) for nome in SCHEMA_TYPES
        ],
    )


@asynccontextmanager
async def create_checkpointer(db_path: str):
    async with aiosqlite.connect(db_path) as conn:
        saver = AsyncSqliteSaver(conn, serde=build_serializer())
        await saver.setup()
        yield saver


@dataclass(frozen=True)
class OrquestradorConfig:
    max_rodadas: int = 2
    max_entrevistas: int = 2


class Orquestrador:
    def __init__(
        self,
        modelos,
        prompts=None,
        checkpointer=None,
        config: OrquestradorConfig | None = None,
    ):
        # Configura o orquestrador com o modelo, número máximo de rodadas e entrevistas
        cfg = config or OrquestradorConfig()
        self.max_rodadas = cfg.max_rodadas
        self.max_entrevistas = cfg.max_entrevistas
        prompts = prompts or {}

        def _modelo_para(papel: str):
            return modelos[papel] if isinstance(modelos, dict) else modelos

        self.analista = Analista(
            _modelo_para('interviewer'), prompts.get('interviewer')
        )
        self.especialistas = {
            nome: Especialista(nome, _modelo_para(nome), prompts.get(nome))
            for nome in PAPEIS
        }
        self.checkpointer = checkpointer or InMemorySaver(
            serde=build_serializer()
        )
        self.grafo = self._construir_grafo()

    def _construir_grafo(self):
        # Cria o grafo de estados do projeto, definindo os nós e as transições entre eles
        grafo = StateGraph(EstadoProjeto)

        # Adiciona os nós do grafo
        grafo.add_node(
            'entrevistador_avaliador', self._mediar
        )  # Nó do entrevistador/avaliador
        grafo.add_node(
            'cliente', self._cliente
        )  # Nó do cliente (interação com o usuário)
        grafo.add_node(
            'resposta', self._finalizar
        )  # Nó de finalização do projeto, retornando o resultado final
        for nome in self.especialistas:
            grafo.add_node(
                nome, partial(self._especialista, nome)
            )  # Nó de cada especialista, chamando a função de avaliação e debate correspondente

        # Define que o nó inicial é o entrevistador/avaliador
        grafo.add_edge(START, 'entrevistador_avaliador')

        # Define as transições condicionais entre os nós:
        # O entrevistador/avaliador pode ir para o cliente, para os especialistas ou finalizar o projeto
        grafo.add_conditional_edges(
            'entrevistador_avaliador',
            self._proximo,
            path_map=['cliente', 'resposta', *self.especialistas],
        )

        # Se o próximo nó for o cliente, ele interage com o usuário e retorna ao entrevistador/avaliador
        grafo.add_conditional_edges(
            'cliente',
            self._proximo_apos_cliente,
            path_map=['cliente', 'entrevistador_avaliador'],
        )

        # O mediador só continua depois que os três especialistas terminarem.
        grafo.add_edge(list(self.especialistas), 'entrevistador_avaliador')

        # O nó de finalização retorna o resultado do projeto e encerra o grafo
        grafo.add_edge('resposta', END)

        return grafo.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _relato(estado):
        # Constrói o relato completo do cliente, incluindo a descrição inicial e as respostas das entrevistas
        relato = estado['descricao']
        for entrevista in estado.get('entrevistas', []):
            for pergunta, resposta in zip(
                entrevista.perguntas, entrevista.respostas
            ):
                texto = resposta or '(não respondeu)'
                relato += (
                    f'\n\nPergunta: {pergunta}\nResposta do cliente: {texto}'
                )
        return relato

    async def _mediar(self, estado):
        # Recupera a fase atual do projeto
        fase = estado.get('fase', 'inicio')

        # Se a fase for inicial
        if fase == 'inicio':
            # Executa a primeira entrevista com o cliente, gerando a especificação inicial e as perguntas a serem feitas
            entrevista = await self.analista.entrevistar(estado['descricao'])
            return {
                'especificacao': entrevista.especificacao,
                'perguntas': entrevista.perguntas,
                'respostas_atuais': [],
                'fase': 'avaliacao',
                'numero_rodada': 1,
                'pareceres': Overwrite({}),
                'debates': Overwrite({}),
                'rodadas': [],
                'entrevistas': [],
                'cliente_encerrado': False,
                'consenso': False,
                'acao': 'consultar',
            }

        # Se a fase for do cliente
        if fase == 'cliente':
            # Se o cliente já encerrou a entrevista e não houve nenhuma interação, passa para a fase de avaliação
            if estado['cliente_encerrado'] and not estado['entrevistas']:
                return {'fase': 'avaliacao', 'acao': 'avaliar'}

            # Realiza uma nova entrevista com o cliente, incorporando as respostas anteriores
            entrevista = await self.analista.entrevistar(
                self._relato(estado),
                estado['especificacao'],
            )

            perguntas = entrevista.perguntas

            # Se o cliente já encerrou a entrevista, mantém as perguntas anteriores e adiciona as novas, removendo duplicatas
            if estado['cliente_encerrado']:
                perguntas = list(
                    dict.fromkeys([*estado['perguntas'], *perguntas])
                )
            return {
                'especificacao': entrevista.especificacao,
                'perguntas': perguntas,
                'respostas_atuais': [],
                'fase': 'avaliacao',
                'acao': 'consultar',
            }

        # Se a fase for de avaliação
        if fase == 'avaliacao':
            consenso = self.aprovado_tecnicamente(estado['pareceres'])

            # Verifica se houve consenso entre os especialistas ou se o número máximo de rodadas foi atingido
            encerrar = consenso or estado['numero_rodada'] > self.max_rodadas
            return {
                'consenso': consenso,
                'debates': Overwrite({}),
                'fase': 'debate',
                'acao': 'finalizar' if encerrar else 'debater',
            }

        # Consolida as decisões do debate e atualiza a especificação com base nos pareceres e debates dos especialistas
        consolidacao = await self.analista.consolidar(
            self._relato(estado),
            estado['especificacao'],
            estado['pareceres'],
            estado['debates'],
        )
        registro = RegistroRodada(
            numero=estado['numero_rodada'],
            pareceres=estado['pareceres'],
            debates=estado['debates'],
            decisoes=consolidacao.decisoes,
        )
        return {
            'especificacao': consolidacao.especificacao,
            'perguntas': consolidacao.perguntas_cliente,
            'rodadas': [*estado['rodadas'], registro],
            'numero_rodada': estado['numero_rodada'] + 1,
            'pareceres': Overwrite({}),
            'respostas_atuais': [],
            'fase': 'avaliacao',
            'acao': 'consultar',
        }

    # Determina o próximo nó do grafo com base na ação atual e no estado do projeto
    def _proximo(self, estado):
        # Se a ação for finalizar, retorna o nó de resposta
        if estado['acao'] == 'finalizar':
            return 'resposta'

        # Se a ação for consultar e houver perguntas pendentes, direciona para o nó do cliente
        if (
            estado['acao'] == 'consultar'
            and estado['perguntas']
            and not estado['cliente_encerrado']
            and len(estado['entrevistas']) < self.max_entrevistas
        ):
            return 'cliente'

        # Se a ação for debater, direciona para o nó do especialista correspondente
        return list(self.especialistas)

    @staticmethod
    def _proximo_apos_cliente(estado):
        if not estado['cliente_encerrado'] and len(
            estado['respostas_atuais']
        ) < len(estado['perguntas']):
            return 'cliente'
        return 'entrevistador_avaliador'

    @staticmethod
    def _cliente(estado):
        perguntas = estado['perguntas']
        respostas_atuais = estado.get('respostas_atuais', [])
        indice = len(respostas_atuais)
        pergunta_atual = perguntas[indice]

        # Solicita ao usuário as respostas para as perguntas pendentes
        retorno = interrupt({
            'pergunta': pergunta_atual,
            'indice': indice + 1,
            'total': len(perguntas),
        })

        # Verifica se o cliente encerrou a entrevista e obtém as respostas fornecidas
        encerrado = (
            isinstance(retorno, dict) and retorno.get('encerrar') is True
        )
        resposta_texto = retorno.get('resposta', '') if encerrado else retorno

        if not isinstance(resposta_texto, str):
            raise ValueError('A resposta deve ser um texto')

        texto_limpo = resposta_texto.strip()
        if not encerrado or texto_limpo:
            respostas_atuais = [*respostas_atuais, texto_limpo]
        lote_concluido = encerrado or len(respostas_atuais) >= len(perguntas)

        if not lote_concluido:
            return {
                'respostas_atuais': respostas_atuais,
                'cliente_encerrado': False,
            }

        entrevistas = estado['entrevistas']

        # Se houver respostas fornecidas, cria uma nova interação com o cliente e adiciona à lista de entrevistas
        if respostas_atuais:
            respondidas = perguntas[: len(respostas_atuais)]
            entrevistas = [
                *entrevistas,
                InteracaoCliente(
                    perguntas=respondidas,
                    respostas=respostas_atuais,
                ),
            ]

        return {
            'entrevistas': entrevistas,
            'perguntas': estado['perguntas'][len(respostas_atuais) :],
            'respostas_atuais': [],
            'cliente_encerrado': encerrado,
            'fase': 'cliente',
        }

    # Executa a avaliação e o debate do especialista, retornando os pareceres ou debates gerados
    async def _especialista(self, nome, estado):
        # Recupera o especialista correspondente ao nome fornecido e o relato do estado atual do projeto
        especialista = self.especialistas[nome]
        relato = self._relato(estado)

        # Se a ação for debater, chama o método de debate do especialista e retorna os debates gerados
        if estado['acao'] == 'debater':
            debate = await especialista.debater(
                relato,
                estado['especificacao'],
                estado['pareceres'],
            )
            return {'debates': {nome: debate}}

        # Se a ação for avaliar, chama o método de avaliação do especialista e retorna os pareceres gerados
        mensagens = [
            {'remetente': remetente, **mensagem.model_dump()}
            for remetente, debate in estado['debates'].items()
            for mensagem in debate.mensagens
            if mensagem.destinatario == nome
        ]
        parecer = await especialista.avaliar(
            relato, estado['especificacao'], mensagens
        )
        return {'pareceres': {nome: parecer}}

    @staticmethod
    def aprovado_tecnicamente(pareceres):
        # Verifica se todos os especialistas aprovaram a especificação e se não há observações críticas
        return (
            set(pareceres) == set(PAPEIS)
            and all(p.aprovado for p in pareceres.values())
            and not any(
                obs.gravidade in {'alta', 'critica'}
                for parecer in pareceres.values()
                for obs in parecer.observacoes
            )
        )

    # Finaliza o projeto, consolidando os resultados e retornando o estado final
    def _finalizar(self, estado):
        rodadas = estado['rodadas']

        # Se o número da rodada atual for menor ou igual ao máximo permitido, adiciona uma nova rodada ao histórico
        if estado['numero_rodada'] <= self.max_rodadas:
            rodadas = [
                *rodadas,
                RegistroRodada(
                    numero=estado['numero_rodada'],
                    pareceres=estado['pareceres'],
                    debates={},
                    decisoes=[],
                ),
            ]

        # Cria o resultado final do projeto, incluindo a descrição do cliente, status, especificação, rodadas, pareceres finais e entrevistas realizadas
        especificacao = estado['especificacao']
        pendencias = list(
            dict.fromkeys([
                *especificacao.questoes_em_aberto,
                *estado['perguntas'],
            ])
        )
        especificacao = especificacao.model_copy(
            update={'questoes_em_aberto': pendencias}
        )
        aprovado = (
            estado['consenso']
            and not pendencias
            and not estado['cliente_encerrado']
        )
        resultado = ResultadoProjeto(
            descricao_cliente=estado['descricao'],
            status='aprovado_tecnicamente' if aprovado else 'com_pendencias',
            especificacao=especificacao,
            rodadas=rodadas,
            pareceres_finais=estado['pareceres'],
            entrevistas=estado['entrevistas'],
        )
        return {
            'resultado': resultado,
            'rodadas': rodadas,
            'especificacao': especificacao,
        }

    def _config(self, thread_id: str) -> dict:
        recursion_limit = 6 * self.max_rodadas + 10 * self.max_entrevistas + 15
        return {
            'configurable': {'thread_id': thread_id},
            'recursion_limit': recursion_limit,
        }

    @staticmethod
    def _formatar_turno(estado: dict) -> dict:
        if estado.get('__interrupt__'):
            dados = estado['__interrupt__'][0].value
            return {
                'status': 'waiting_user',
                'pergunta': dados['pergunta'],
                'pergunta_indice': dados['indice'],
                'pergunta_total': dados['total'],
                'resultado': None,
            }
        resultado: ResultadoProjeto = estado['resultado']
        status = (
            'completed'
            if resultado.status == 'aprovado_tecnicamente'
            else 'completed_with_pending'
        )
        return {
            'status': status,
            'pergunta': None,
            'pergunta_indice': None,
            'pergunta_total': None,
            'resultado': resultado,
        }

    async def iniciar(self, thread_id: str, descricao: str) -> dict:
        texto = descricao.strip()
        if not texto:
            raise ValueError('A descrição do projeto não pode estar vazia')
        estado = await self.grafo.ainvoke(
            {'descricao': texto},
            config=self._config(thread_id),
        )
        return self._formatar_turno(estado)

    async def responder(
        self, thread_id: str, resposta: str, *, encerrar: bool = False
    ) -> dict:
        # Executa o orquestrador, iniciando o processo de entrevista, avaliação e consolidação do projeto
        texto = resposta.strip()
        eh_encerramento = encerrar or texto.lower() == '/encerrar'

        if eh_encerramento:
            payload = {
                'encerrar': True,
                'resposta': '' if texto.lower() == '/encerrar' else texto,
            }
        else:
            payload = texto

        estado = await self.grafo.ainvoke(
            Command(resume=payload),
            config=self._config(thread_id),
        )

        return self._formatar_turno(estado)

    async def stream(
        self,
        thread_id: str,
        texto: str,
        *,
        iniciar: bool,
        encerrar: bool = False,
    ):
        desc_limpa = texto.strip()

        if iniciar:
            entrada = {'descricao': desc_limpa}
        else:
            fechar = encerrar or desc_limpa.lower() == '/encerrar'

            if fechar:
                payload = {
                    'encerrar': True,
                    'resposta': '' if desc_limpa.lower() == '/encerrar' else desc_limpa,
                }
            else:
                payload = desc_limpa
    
            entrada = Command(resume=payload)

        estado = None

        async for modo, chunk in self.grafo.astream(
            entrada,
            config=self._config(thread_id),
            stream_mode=['updates', 'values'], # Verifica o estado completo (values) e também os atualizações parciais (updates)
        ):
            if modo == 'values':
                estado = chunk
            elif modo == 'updates':
                for status_update in chunk:
                    if status_update in STATUS_BOTS:
                        yield {'tipo': 'status', 'mensagem': STATUS_BOTS[status_update]}

        yield {'tipo': 'done', 'turno': self._formatar_turno(estado)}
