"""Enriquecimento da base com fontes externas (IBGE e Atlas do Desenvolvimento Humano).

As fontes são baixadas uma única vez de repositórios públicos que redistribuem os dados oficiais
e salvas em ``data/external`` (versionadas no Git), para que a pipeline rode sem internet.

| Arquivo | Fonte oficial | Redistribuição utilizada |
|---|---|---|
| ``municipios_ibge.csv`` | IBGE – códigos, coordenadas e capitais | github.com/kelvins/municipios-brasileiros (MIT) |
| ``populacao_ibge.csv`` | IBGE – Censos 2010/2022 e estimativas anuais (SIDRA/DOU) | pacote R ``brpop`` (CRAN) |
| ``atlas_idhm_2010.csv`` | Atlas do Desenvolvimento Humano no Brasil (PNUD/IPEA/FJP), Censo 2010 | pacote R ``abjData`` (CRAN) |
"""

from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from src.config import EXTERNAL_DIR

URLS = {
    "municipios": "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv",
    "populacao": "https://raw.githubusercontent.com/cran/brpop/master/inst/extdata/ibge_pop.rds",
    "atlas": "https://raw.githubusercontent.com/abjur/abjData/master/data/pnud_muni.rda",
}

# Indicadores do Atlas (Censo 2010) selecionados por relação plausível com alfabetização infantil
ATLAS_COLS = {
    "idhm": "idhm",                       # IDH Municipal
    "idhm_e": "idhm_educacao",            # dimensão Educação
    "idhm_l": "idhm_longevidade",         # dimensão Longevidade
    "idhm_r": "idhm_renda",               # dimensão Renda
    "rdpc": "renda_per_capita",           # renda domiciliar per capita (R$ de 2010)
    "gini": "gini",                       # desigualdade de renda
    "pind": "pct_extrema_pobreza",        # % extremamente pobres
    "ppob": "pct_pobres",                 # % pobres
    "pindcri": "pct_criancas_extrema_pobreza",
    "t_analf18m": "taxa_analfabetismo_18mais",
    "e_anosestudo": "anos_estudo_esperados",
    "t_freq4a5": "taxa_freq_escola_4a5",  # acesso à pré-escola
    "t_freq5a6": "taxa_freq_escola_5a6",
    "t_atraso_2_fund": "taxa_atraso_2anos_fund",  # distorção idade-série
    "t_crifundin_todos": "pct_criancas_lar_sem_fund_completo",  # nenhum morador com fundamental
    "t_agua": "pct_agua_encanada",
    "agua_esgoto": "pct_sem_agua_esgoto_adequado",
    "t_luz": "pct_energia_eletrica",
    "mort5": "mortalidade_ate_5anos",
    "espvida": "esperanca_vida",
    "razdep": "razao_dependencia",
    "p_formal": "grau_formalizacao_trabalho",
    "t_des": "taxa_desocupacao",
}


def _baixar(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (URLs fixas e públicas)
        return r.read()


def baixar_fontes_externas(forcar: bool = False) -> None:
    """Baixa e normaliza as fontes externas para CSVs em data/external (executar uma vez)."""
    destino = {
        "municipios": EXTERNAL_DIR / "municipios_ibge.csv",
        "populacao": EXTERNAL_DIR / "populacao_ibge.csv",
        "atlas": EXTERNAL_DIR / "atlas_idhm_2010.csv",
    }
    if not forcar and all(p.exists() for p in destino.values()):
        return

    # Importados só aqui: quem já tem os CSVs em data/external não precisa do pyreadr.
    import tempfile

    import pyreadr  # dependência apenas para a etapa de download

    mun = pd.read_csv(io.BytesIO(_baixar(URLS["municipios"])))
    mun = mun.rename(columns={"codigo_ibge": "id_municipio", "nome": "nome_municipio"})
    mun[["id_municipio", "nome_municipio", "latitude", "longitude", "capital", "codigo_uf"]].to_csv(
        destino["municipios"], index=False)

    with tempfile.TemporaryDirectory() as tmp:
        for chave, nome in [("populacao", "ibge_pop.rds"), ("atlas", "pnud_muni.rda")]:
            caminho = f"{tmp}/{nome}"
            with open(caminho, "wb") as f:
                f.write(_baixar(URLS[chave]))
            df = next(iter(pyreadr.read_r(caminho).values()))
            if chave == "populacao":
                df = df.rename(columns={"code_muni": "id_municipio", "year": "ano", "pop": "populacao"})
                df = df.dropna(subset=["populacao"]).astype({"id_municipio": int, "ano": int, "populacao": int})
                df = df[df["ano"] >= 2010]
                df.sort_values(["id_municipio", "ano"]).to_csv(destino["populacao"], index=False)
            else:
                df = df[df["ano"] == 2010]
                df = df[["codmun7", "pesotot", "pesourb"] + list(ATLAS_COLS)].rename(
                    columns={"codmun7": "id_municipio", "pesotot": "pop_2010", "pesourb": "pop_urbana_2010",
                             **ATLAS_COLS})
                df["id_municipio"] = df["id_municipio"].astype(int)
                df.to_csv(destino["atlas"], index=False)


def construir_features_municipio() -> pd.DataFrame:
    """Tabela única com atributos territoriais e socioeconômicos por município (1 linha/município)."""
    mun = pd.read_csv(EXTERNAL_DIR / "municipios_ibge.csv")
    pop = pd.read_csv(EXTERNAL_DIR / "populacao_ibge.csv")
    atlas = pd.read_csv(EXTERNAL_DIR / "atlas_idhm_2010.csv")

    pop_w = pop.pivot(index="id_municipio", columns="ano", values="populacao")
    feats = pd.DataFrame({
        "id_municipio": pop_w.index,
        "populacao_2022": pop_w[2022].values,
        "crescimento_pop_2010_2022": (pop_w[2022] / pop_w[2010] - 1).values,
    })

    df = (mun[["id_municipio", "nome_municipio", "latitude", "longitude", "capital"]]
          .merge(feats, on="id_municipio", how="left")
          .merge(atlas, on="id_municipio", how="left"))

    df["log_populacao"] = np.log10(df["populacao_2022"])
    df["taxa_urbanizacao_2010"] = df["pop_urbana_2010"] / df["pop_2010"]
    df["porte_municipio"] = pd.cut(
        df["populacao_2022"], bins=[0, 5_000, 20_000, 100_000, 500_000, np.inf],
        labels=["Até 5 mil", "5–20 mil", "20–100 mil", "100–500 mil", "Mais de 500 mil"]).astype(object)
    return df.drop(columns=["pop_urbana_2010", "pop_2010"])
