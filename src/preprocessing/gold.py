"""Reconstrução das camadas Silver e Gold da Fase 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CORTE_ALFABETIZACAO, MAPA_REDE, PROCESSED_DIR, RAW_DIR, RAW_FILES

CODIGO_UF = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA",
    31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS",
    50: "MS", 51: "MT", 52: "GO", 53: "DF",
}
REGIAO_UF = {
    1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste",
}

METAS = [f"meta_alfabetizacao_{a}" for a in range(2024, 2031)]


# --------------------------------------------------------------------------------------
# Bronze -> Silver
# --------------------------------------------------------------------------------------
def carregar_bronze() -> dict[str, pd.DataFrame]:
    """Lê os seis CSVs originais sem alterações."""
    return {nome: pd.read_csv(RAW_DIR / arq) for nome, arq in RAW_FILES.items()}


def _adicionar_territorio(df: pd.DataFrame) -> pd.DataFrame:
    cod_uf = (df["id_municipio"] // 100_000).astype(int)
    df = df.copy()
    df["sigla_uf"] = cod_uf.map(CODIGO_UF)
    df["regiao"] = (cod_uf // 10).map(REGIAO_UF)
    return df


def construir_silver(bronze: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Aplica as padronizações da Silver: tipos, textos, duplicidades e domínios."""
    silver = {}
    for nome, df in bronze.items():
        df = df.copy().drop_duplicates()
        if "sigla_uf" in df.columns:
            df["sigla_uf"] = df["sigla_uf"].str.strip().str.upper()
        if "rede" in df.columns and pd.api.types.is_numeric_dtype(df["rede"]):
            df["rede_ensino"] = df["rede"].map(MAPA_REDE).fillna("Desconhecida")
        silver[nome] = df

    alunos = silver["avaliacao_alunos"]
    for c in ["presenca", "preenchimento_caderno", "alfabetizado"]:
        alunos[c] = alunos[c].astype("int8")
    alunos = _adicionar_territorio(alunos)
    silver["avaliacao_alunos"] = alunos

    silver["avaliacao_municipio"] = _adicionar_territorio(silver["avaliacao_municipio"])
    silver["meta_municipio"] = _adicionar_territorio(silver["meta_municipio"])
    return silver


def validar_silver(silver: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Regras de qualidade"""
    a = silver["avaliacao_alunos"]
    validos = a[a["preenchimento_caderno"] == 1]
    regras = [
        ("alunos: id_aluno+ano único", not a.duplicated(["ano", "id_aluno"]).any()),
        ("alunos: binários em {0,1}",
         a[["presenca", "preenchimento_caderno", "alfabetizado"]].isin([0, 1]).all().all()),
        ("alunos: caderno não preenchido => sem proficiência",
         a.loc[a["preenchimento_caderno"] == 0, "proficiencia"].isna().all()),
        ("alunos: alfabetizado == (proficiência >= 743)",
         (validos["alfabetizado"] == (validos["proficiencia"] >= CORTE_ALFABETIZACAO)).all()),
        ("alunos: caderno não preenchido => alfabetizado = 0",
         (a.loc[a["preenchimento_caderno"] == 0, "alfabetizado"] == 0).all()),
        ("município: taxa em [0,100]",
         silver["avaliacao_municipio"]["taxa_alfabetizacao"].between(0, 100).all()),
        ("metas município: metas em [0,100]",
         silver["meta_municipio"][METAS].stack().dropna().between(0, 100).all()),
        ("município: ano+município+rede único",
         not silver["avaliacao_municipio"].duplicated(["ano", "id_municipio", "rede"]).any()),
        ("metas município: ano+município único",
         not silver["meta_municipio"].duplicated(["ano", "id_municipio"]).any()),
    ]
    return pd.DataFrame(regras, columns=["regra", "ok"]).assign(
        status=lambda d: np.where(d["ok"], "PASS", "FAIL"))


# --------------------------------------------------------------------------------------
# Silver -> Gold
# --------------------------------------------------------------------------------------
def gold_brasil(silver):
    df = silver["meta_brasil"].copy()
    df["meta_ano"] = np.where(df["ano"] == 2024, df["meta_alfabetizacao_2024"],
                              np.where(df["ano"] == 2025, df["meta_alfabetizacao_2025"], np.nan))
    df["diferenca_para_meta"] = (df["taxa_alfabetizacao"] - df["meta_ano"]).round(2)
    df["status_meta"] = np.select(
        [df["meta_ano"].isna(), df["taxa_alfabetizacao"] >= df["meta_ano"]],
        ["SEM_META", "ATINGIDA"], "NAO_ATINGIDA")
    return df.sort_values("ano").reset_index(drop=True)


def gold_uf(silver):
    cols = ["ano", "sigla_uf", "rede", "rede_ensino", "serie", "taxa_alfabetizacao", "media_portugues"]
    return silver["avaliacao_uf"][cols].sort_values(["ano", "sigla_uf", "rede"]).reset_index(drop=True)


def gold_municipio(silver):
    """Resultado municipal (rede Municipal, compatível com as metas municipais) + metas."""
    aval = silver["avaliacao_municipio"]
    municipal = aval[aval["rede"] == 3][
        ["ano", "id_municipio", "sigla_uf", "regiao", "taxa_alfabetizacao", "media_portugues"]]
    meta = silver["meta_municipio"][
        ["ano", "id_municipio", "percentual_participacao", "nivel_alfabetizacao"] + METAS]
    df = municipal.merge(meta, on=["ano", "id_municipio"], how="left")
    df["meta_ano"] = np.where(df["ano"] == 2024, df["meta_alfabetizacao_2024"], np.nan)
    df["diferenca_para_meta"] = (df["taxa_alfabetizacao"] - df["meta_ano"]).round(2)
    df["status_meta"] = np.select(
        [df["meta_ano"].isna(), df["taxa_alfabetizacao"] >= df["meta_ano"]],
        ["SEM_META", "ATINGIDA"], "NAO_ATINGIDA")
    return df.sort_values(["ano", "id_municipio"]).reset_index(drop=True)


def gold_municipio_evolucao(gold_mun):
    p = gold_mun.pivot_table(index=["id_municipio", "sigla_uf", "regiao"], columns="ano",
                             values=["taxa_alfabetizacao", "media_portugues"])
    p.columns = [f"{v}_{a}" for v, a in p.columns]
    p = p.reset_index()
    p["variacao_taxa"] = (p["taxa_alfabetizacao_2024"] - p["taxa_alfabetizacao_2023"]).round(2)
    p["variacao_media_portugues"] = (p["media_portugues_2024"] - p["media_portugues_2023"]).round(2)
    p["tendencia"] = np.select([p["variacao_taxa"] > 0, p["variacao_taxa"] < 0],
                               ["MELHORA", "PIORA"], "ESTAVEL")
    p.loc[p["variacao_taxa"].isna(), "tendencia"] = "SEM_COMPARACAO"
    return p


def gold_municipio_redes(silver):
    """Visão larga por município: taxa e média de cada rede (Estadual, Municipal, Pública)."""
    aval = silver["avaliacao_municipio"]
    aval = aval[aval["rede"].isin([2, 3, 5])]
    p = aval.pivot_table(index=["ano", "id_municipio"], columns="rede_ensino",
                         values=["taxa_alfabetizacao", "media_portugues"])
    p.columns = [f"{v}_{r.lower().replace('ú', 'u')}" for v, r in p.columns]
    return p.reset_index()


def gold_alunos(silver):
    """Base de alunos (nível do indivíduo) com flags de avaliação válida."""
    a = silver["avaliacao_alunos"].copy()
    a["avaliacao_valida"] = (a["preenchimento_caderno"] == 1).astype("int8")
    return a


def gold_indicadores_alunos(alunos):
    return (alunos.groupby(["ano", "rede_ensino"])
            .agg(alunos=("id_aluno", "count"),
                 taxa_presenca=("presenca", "mean"),
                 avaliacoes_validas=("avaliacao_valida", "sum"),
                 taxa_alfabetizacao_valida=("alfabetizado",
                                            lambda s: s[alunos.loc[s.index, "avaliacao_valida"] == 1].mean()),
                 proficiencia_media=("proficiencia", "mean"))
            .reset_index())


def construir_gold(salvar: bool = True) -> dict[str, pd.DataFrame]:
    silver = construir_silver(carregar_bronze())
    gm = gold_municipio(silver)
    alunos = gold_alunos(silver)
    gold = {
        "gold_brasil_desempenho": gold_brasil(silver),
        "gold_uf_desempenho": gold_uf(silver),
        "gold_municipio_desempenho": gm,
        "gold_municipio_evolucao": gold_municipio_evolucao(gm),
        "gold_municipio_redes": gold_municipio_redes(silver),
        "gold_alunos": alunos,
        "gold_indicadores_alunos": gold_indicadores_alunos(alunos),
    }
    if salvar:
        for nome, df in gold.items():
            df.to_parquet(PROCESSED_DIR / f"{nome}.parquet", index=False)
    return gold


def carregar_gold(nome: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{nome}.parquet")
