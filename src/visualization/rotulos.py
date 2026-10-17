"""Nomes legíveis das variáveis, para usar nos gráficos e tabelas apresentados.

Use ``rotulo(nome)`` para converter um nome técnico de coluna (inclusive as colunas criadas pelo One-Hot,
como ``sigla_uf_CE``) no nome que aparece para o leitor.
"""

from __future__ import annotations

ROTULOS = {
    # Aluno e território
    "rede_ensino": "Rede de ensino", "sigla_uf": "UF", "regiao": "Região", "porte_municipio": "Porte do município",
    "capital": "Capital", "latitude": "Latitude", "longitude": "Longitude", "log_populacao": "População (log)",
    "crescimento_pop_2010_2022": "Crescimento populacional 2010–2022",
    "taxa_urbanizacao_2010": "Taxa de urbanização (2010)",
    # Socioeconômicas (Atlas, Censo 2010)
    "idhm": "IDHM", "idhm_educacao": "IDHM Educação", "idhm_longevidade": "IDHM Longevidade", "idhm_renda": "IDHM Renda",
    "renda_per_capita": "Renda per capita", "log_renda_per_capita": "Renda per capita (log)",
    "gini": "Desigualdade de renda (Gini)", "pct_extrema_pobreza": "% em extrema pobreza", "pct_pobres": "% de pobres",
    "pct_criancas_extrema_pobreza": "% de crianças em extrema pobreza",
    "taxa_analfabetismo_18mais": "Analfabetismo (18 anos ou mais)", "anos_estudo_esperados": "Anos de estudo esperados",
    "taxa_freq_escola_4a5": "Frequência escolar (4 a 5 anos)", "taxa_freq_escola_5a6": "Frequência escolar (5 a 6 anos)",
    "taxa_atraso_2anos_fund": "Atraso escolar de 2+ anos (fundamental)",
    "pct_criancas_lar_sem_fund_completo": "Crianças em lar sem fundamental completo",
    "pct_agua_encanada": "% com água encanada", "pct_sem_agua_esgoto_adequado": "% sem água e esgoto adequados",
    "pct_energia_eletrica": "% com energia elétrica", "mortalidade_ate_5anos": "Mortalidade até 5 anos",
    "esperanca_vida": "Esperança de vida", "razao_dependencia": "Razão de dependência",
    "grau_formalizacao_trabalho": "Formalização do trabalho", "taxa_desocupacao": "Taxa de desocupação",
    # Histórico educacional do ano anterior (modelo de alunos)
    "taxa_rede_aluno_ant": "Taxa da rede do aluno (2023)",
    "media_rede_aluno_ant": "Média de Português da rede do aluno (2023)",
    "taxa_alfabetizacao_estadual_ant": "Taxa da rede estadual (2023)",
    "taxa_alfabetizacao_municipal_ant": "Taxa da rede municipal (2023)",
    "taxa_alfabetizacao_publica_ant": "Taxa da rede pública (2023)",
    "media_portugues_estadual_ant": "Média de Português – rede estadual (2023)",
    "media_portugues_municipal_ant": "Média de Português – rede municipal (2023)",
    "media_portugues_publica_ant": "Média de Português – rede pública (2023)",
    "participacao_ant": "Participação na avaliação (2023)", "nivel_alfabetizacao_ant": "Nível de alfabetização (2023)",
    "taxa_uf_publica_ant": "Taxa da UF – rede pública (2023)", "tem_rede_estadual": "Tem rede estadual",
    "dif_estadual_municipal_ant": "Diferença estadual − municipal (2023)",
    "gap_rede_aluno_vs_uf_ant": "Distância da rede do aluno para a UF (2023)",
    # Metas (modelo de alunos)
    **{f"meta_alfabetizacao_{a}": f"Meta municipal {a}" for a in range(2024, 2031)},
    "esforco_meta_2024": "Esforço exigido pela meta 2024",
    # Desempenho atual e metas (modelo municipal)
    "taxa_atual": "Taxa atual do município", "media_atual": "Média de Português atual",
    "participacao_atual": "Participação na avaliação (atual)", "nivel_atual": "Nível de alfabetização (atual)",
    "taxa_estadual_atual": "Taxa da rede estadual no município (atual)",
    "media_estadual_atual": "Média de Português – rede estadual (atual)",
    "taxa_uf_publica_atual": "Taxa da UF – rede pública (atual)", "meta_alvo": "Meta do ano seguinte",
    "esforco_meta": "Esforço exigido pela meta", "distancia_meta_2030": "Distância para a meta de 80% (2030)",
    "gap_para_uf": "Distância do município para a UF",
}

_CATEGORICAS = ["rede_ensino", "sigla_uf", "regiao", "porte_municipio"]


def rotulo(nome: str) -> str:
    """Nome legível de uma coluna. Colunas do One-Hot viram "Variável: categoria" (ex.: "UF: CE")."""
    nome = nome.replace("missingindicator_", "")
    if nome in ROTULOS:
        return ROTULOS[nome]
    for c in sorted(_CATEGORICAS, key=len, reverse=True):
        if nome.startswith(c + "_"):
            categoria = nome[len(c) + 1:].replace("infrequent_sklearn", "outras")
            return f"{ROTULOS[c]}: {categoria}"
    return nome
