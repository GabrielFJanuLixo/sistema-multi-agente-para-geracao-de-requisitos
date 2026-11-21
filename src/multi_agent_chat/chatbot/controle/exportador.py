from multi_agent_chat.chatbot.estados.schemas import ResultadoProjeto


def formatar_pergunta_markdown(pergunta: str, indice: int, total: int):
    return (
        f'**Pergunta {indice} de {total}:**\n\n'
        f'{pergunta}\n\n'
        '*(Envie `/encerrar` se quiser seguir com as informações atuais.)*'
    )


def formatar_especificacao_markdown(resultado: ResultadoProjeto) -> str:
    linhas = [
        '# Especificação de Requisitos de Software',
        '',
        f'**Status da revisão:** {resultado.status}',
        '',
        '## Descrição original',
        resultado.descricao_cliente,
        '',
        '## Requisitos',
    ]
    for req in resultado.especificacao.requisitos:
        linhas.extend([
            '',
            f'### {req.id} — {req.categoria} ({req.origem})',
            req.descricao,
            f'**Evidência/justificativa:** {req.evidencia}',
            '**Critérios de aceitação:**',
        ])
        linhas.extend(f'- {criterio}' for criterio in req.criterios_aceitacao)
    linhas.extend(['', '## Sugestões não confirmadas'])
    linhas.extend(
        f'- {item}'
        for item in resultado.especificacao.sugestoes_nao_confirmadas
    )
    linhas.extend(['', '## Questões em aberto'])
    linhas.extend(
        f'- {item}' for item in resultado.especificacao.questoes_em_aberto
    )
    linhas.extend(['', '## Pendências técnicas da revisão final'])
    for nome, parecer in resultado.pareceres_finais.items():
        for obs in parecer.observacoes:
            linhas.append(
                f'- **{nome} / {obs.requisito_id} ({obs.gravidade})**: '
                f'{obs.problema} — {obs.recomendacao}'
            )
    linhas.extend(['', '## Conversa com o cliente'])
    for entrevista in resultado.entrevistas:
        for pergunta, resposta in zip(
            entrevista.perguntas, entrevista.respostas
        ):
            linhas.extend([
                '',
                f'**Pergunta:** {pergunta}',
                f'**Resposta:** {resposta or "Não respondida."}',
            ])
    return '\n'.join(linhas) + '\n'
