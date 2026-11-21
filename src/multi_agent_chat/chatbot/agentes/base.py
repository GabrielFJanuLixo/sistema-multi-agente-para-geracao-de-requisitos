from langchain_core.prompts import ChatPromptTemplate


class AgenteEstruturado:
    def __init__(self, modelo, instrucao, schema):
        prompt = ChatPromptTemplate.from_messages([
            ('system', instrucao),
            (
                'human',
                '{tarefa}\n\nDescrição do cliente:\n{descricao}\n\nContexto:\n{contexto}',
            ),
        ])

        # Configura o pipeline de execução: prompt + estrutura de saída do modelo
        self.pipeline = prompt | modelo.with_structured_output(
            schema,
            method='json_schema',
        )

    async def executar(self, tarefa, descricao, contexto=None):
        # Recebe a tarefa, a descrição do cliente e cria o contexto textual
        contexto_texto = '\n'.join(
            f'{nome}: {valor}' for nome, valor in (contexto or {}).items()
        )

        # Invoca o pipeline com a tarefa, descrição e contexto, retornando a resposta estruturada
        return await self.pipeline.ainvoke({
            'tarefa': tarefa,
            'descricao': descricao,
            'contexto': contexto_texto,
        })
