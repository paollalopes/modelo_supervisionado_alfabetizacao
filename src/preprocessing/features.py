"""Construção das bases analíticas usadas na modelagem.

Princípio central: **toda variável explicativa precisa estar disponível ANTES do resultado que se
quer prever.** Por isso os indicadores educacionais do município entram sempre defasados (ano
anterior) e nenhuma informação derivada da prova do próprio aluno é usada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import META_NACIONAL_2030, PROCESSED_DIR
from src.preprocessing.external import construir_features_municipio
from src.preprocessing.gold import carregar_gold

# Variáveis que NUNCA entram no modelo de alunos e o motivo (documentação do tratamento de leakage)
VARIAVEIS_PROIBIDAS = {
    "proficiencia": "É a nota que define o alvo (alfabetizado = nota ≥ 743). Usá-la entregaria a resposta ao modelo.",
    "peso_aluno": "Calculado pelo INEP depois da prova. Não está disponível no momento da previsão.",
    "presenca": "Igual para todos os alunos da base, que só inclui presentes. Não diferencia ninguém.",
    "preenchimento_caderno": "Critério usado para selecionar a base. Igual para todos, sem poder de diferenciação.",
    "id_aluno": "Código de identificação. Não traz nenhuma informação sobre o aluno.",
    "id_escola": "Faria o modelo memorizar escolas em vez de aprender padrões e não funcionaria para escolas novas.",
    "taxa_alfabetizacao (mesmo ano)": "Já inclui o resultado do próprio aluno. O modelo usa a taxa do ano anterior.",
    "nivel_alfabetizacao / percentual_participacao (mesmo ano)": "Vêm da mesma prova que gera o alvo. O modelo usa os valores do ano anterior.",
}


def _indicadores_municipais_defasados(ano_ref: int) -> pd.DataFrame:
    """Indicadores educacionais do município no ano ``ano_ref`` (usados para prever ano_ref + 1)."""
    redes = carregar_gold("gold_municipio_redes")
    redes = redes[redes["ano"] == ano_ref].drop(columns="ano")
    redes = redes.rename(columns={c: f"{c}_ant" for c in redes.columns if c != "id_municipio"})

    gm = carregar_gold("gold_municipio_desempenho")
    gm = gm[gm["ano"] == ano_ref][["id_municipio", "percentual_participacao", "nivel_alfabetizacao"]]
    gm = gm.rename(columns={"percentual_participacao": "participacao_ant",
                            "nivel_alfabetizacao": "nivel_alfabetizacao_ant"})

    uf = carregar_gold("gold_uf_desempenho")
    uf = uf[(uf["ano"] == ano_ref) & (uf["rede"] == 5)][["sigla_uf", "taxa_alfabetizacao"]]
    uf = uf.rename(columns={"taxa_alfabetizacao": "taxa_uf_publica_ant"})

    return redes.merge(gm, on="id_municipio", how="outer"), uf


def _metas_municipio() -> pd.DataFrame:
    """Metas pactuadas (definidas ex ante; idênticas nas linhas de 2023 e 2024)."""
    gm = carregar_gold("gold_municipio_desempenho")
    metas = [f"meta_alfabetizacao_{a}" for a in range(2024, 2031)]
    return gm[gm["ano"] == 2024][["id_municipio"] + metas]


def base_alunos(ano_alvo: int = 2024) -> pd.DataFrame:
    """Base do modelo principal: 1 linha por aluno com avaliação válida em ``ano_alvo``.

    Alvo: ``alfabetizado`` (1 = proficiência >= 743).
    Variáveis: rede do aluno, caderno da prova, território, socioeconomia (IBGE/Atlas),
    metas municipais e resultados do município/UF no ANO ANTERIOR.
    """
    alunos = carregar_gold("gold_alunos")
    alunos = alunos[(alunos["ano"] == ano_alvo) & (alunos["avaliacao_valida"] == 1)]
    alunos = alunos[["id_aluno", "id_escola", "id_municipio", "sigla_uf", "regiao",
                     "rede_ensino", "caderno", "alfabetizado"]].copy()

    mun_ant, uf_ant = _indicadores_municipais_defasados(ano_alvo - 1)
    ext = construir_features_municipio()

    df = (alunos
          .merge(mun_ant, on="id_municipio", how="left")
          .merge(uf_ant, on="sigla_uf", how="left")
          .merge(_metas_municipio(), on="id_municipio", how="left")
          .merge(ext.drop(columns=["nome_municipio"]), on="id_municipio", how="left"))

    # Resultado do ano anterior na MESMA rede do aluno (Estadual ou Municipal)
    df["taxa_rede_aluno_ant"] = np.where(df["rede_ensino"] == "Estadual",
                                         df["taxa_alfabetizacao_estadual_ant"],
                                         df["taxa_alfabetizacao_municipal_ant"])
    df["media_rede_aluno_ant"] = np.where(df["rede_ensino"] == "Estadual",
                                          df["media_portugues_estadual_ant"],
                                          df["media_portugues_municipal_ant"])
    # Esforço exigido pela meta: quanto o município precisa crescer em relação ao ano anterior
    df[f"esforco_meta_{ano_alvo}"] = (df[f"meta_alfabetizacao_{ano_alvo}"]
                                      - df["taxa_alfabetizacao_municipal_ant"])
    df["caderno"] = df["caderno"].astype(str)
    df["capital"] = df["capital"].astype("float")
    return df.reset_index(drop=True)


def base_municipios(ano_ref: int) -> pd.DataFrame:
    """Base municipal: indicadores de ``ano_ref`` para prever a meta de ``ano_ref + 1``.

    - ano_ref=2023 → alvo observado: município NÃO atingiu a meta de 2024 (treino/validação)
    - ano_ref=2024 → alvo desconhecido: risco de não atingir a meta de 2025 (projeção)
    """
    ano_meta = ano_ref + 1
    gm = carregar_gold("gold_municipio_desempenho")
    atual = gm[gm["ano"] == ano_ref][["id_municipio", "sigla_uf", "regiao", "taxa_alfabetizacao",
                                      "media_portugues", "percentual_participacao",
                                      "nivel_alfabetizacao"]]
    atual = atual.rename(columns={"taxa_alfabetizacao": "taxa_atual", "media_portugues": "media_atual",
                                  "percentual_participacao": "participacao_atual",
                                  "nivel_alfabetizacao": "nivel_atual"})

    redes = carregar_gold("gold_municipio_redes")
    redes = redes[redes["ano"] == ano_ref][["id_municipio", "taxa_alfabetizacao_estadual",
                                            "media_portugues_estadual"]]
    redes = redes.rename(columns={"taxa_alfabetizacao_estadual": "taxa_estadual_atual",
                                  "media_portugues_estadual": "media_estadual_atual"})

    uf = carregar_gold("gold_uf_desempenho")
    uf = uf[(uf["ano"] == ano_ref) & (uf["rede"] == 5)][["sigla_uf", "taxa_alfabetizacao"]]
    uf = uf.rename(columns={"taxa_alfabetizacao": "taxa_uf_publica_atual"})

    metas = _metas_municipio()
    ext = construir_features_municipio()

    df = (atual.merge(redes, on="id_municipio", how="left")
          .merge(uf, on="sigla_uf", how="left")
          .merge(metas, on="id_municipio", how="left")
          .merge(ext, on="id_municipio", how="left"))

    df["meta_alvo"] = df[f"meta_alfabetizacao_{ano_meta}"]
    df["esforco_meta"] = df["meta_alvo"] - df["taxa_atual"]           # pontos a crescer
    df["distancia_meta_2030"] = META_NACIONAL_2030 - df["taxa_atual"]
    df["gap_para_uf"] = df["taxa_atual"] - df["taxa_uf_publica_atual"]
    df["capital"] = df["capital"].astype("float")

    if ano_ref == 2023:  # alvo observado
        prox = gm[gm["ano"] == ano_meta][["id_municipio", "taxa_alfabetizacao", "meta_ano", "status_meta"]]
        df = df.merge(prox.rename(columns={"taxa_alfabetizacao": "taxa_proximo_ano"}),
                      on="id_municipio", how="inner")
        df = df[df["status_meta"] != "SEM_META"].copy()
        df["nao_atingiu_meta"] = (df["status_meta"] == "NAO_ATINGIDA").astype(int)

    df = df.drop(columns=[f"meta_alfabetizacao_{a}" for a in range(2024, 2031)])
    df["ano_referencia"] = ano_ref
    return df.reset_index(drop=True)


def salvar_bases() -> dict[str, pd.DataFrame]:
    bases = {
        "base_alunos_2024": base_alunos(2024),
        "base_municipios_2023": base_municipios(2023),
        "base_municipios_2024": base_municipios(2024),
        "features_municipio_externas": construir_features_municipio(),
    }
    for nome, df in bases.items():
        df.to_parquet(PROCESSED_DIR / f"{nome}.parquet", index=False)
    return bases
