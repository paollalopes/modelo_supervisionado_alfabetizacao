"""Modelos no nível do MUNICÍPIO: risco de não atingir a meta e agrupamento por perfil."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE
from src.modeling.pipeline import SOCIOECONOMICAS, TERRITORIO_CAT, TERRITORIO_NUM

# ---------------------------------------------------------------------------------------------
# Modelo de risco de não atingir a meta do ano seguinte
# ---------------------------------------------------------------------------------------------
DESEMPENHO_ATUAL = ["taxa_atual", "media_atual", "participacao_atual", "nivel_atual",
                    "taxa_estadual_atual", "media_estadual_atual", "taxa_uf_publica_atual"]
META = ["meta_alvo", "esforco_meta", "distancia_meta_2030", "gap_para_uf"]
NUM_MUNICIPIO = DESEMPENHO_ATUAL + META + TERRITORIO_NUM + SOCIOECONOMICAS
# UF e região NÃO entram como dummies: com uma única transição (2023 -> 2024) para treinar, os dummies
# memorizam choques específicos daquele ano (ex.: enchentes no RS em 2024) e os projetariam para o futuro.
# O contexto estadual entra de forma contínua (taxa da UF e distância do município para a UF).
CAT_MUNICIPIO = ["porte_municipio"]
CAT_MUNICIPIO_COM_UF = TERRITORIO_CAT
ALVO_MUNICIPIO = "nao_atingiu_meta"

# Colunas que existem na base de treino mas descrevem o PRÓPRIO desfecho (nunca usar como variável)
VAZAMENTO_MUNICIPIO = ["taxa_proximo_ano", "meta_ano", "status_meta", "nao_atingiu_meta"]


# ---------------------------------------------------------------------------------------------
# Agrupamento (clusterização) de municípios
# ---------------------------------------------------------------------------------------------
VARS_CLUSTER = {
    "taxa_2024": "Taxa de alfabetização 2024",
    "variacao_taxa": "Variação 2023→2024 (p.p.)",
    "idhm": "IDHM 2010",
    "renda_per_capita": "Renda per capita 2010",
    "pct_extrema_pobreza": "% extrema pobreza 2010",
    "taxa_atraso_2anos_fund": "Distorção idade-série 2010",
    "taxa_analfabetismo_18mais": "Analfabetismo adulto 2010",
    "log_populacao": "log10 população 2022",
}


def pipeline_cluster(k: int) -> Pipeline:
    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("padronizacao", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)),
    ])


def escolher_k(X: pd.DataFrame, ks=range(2, 9)) -> pd.DataFrame:
    """Inércia (cotovelo) e silhueta para cada k."""
    linhas = []
    for k in ks:
        p = pipeline_cluster(k).fit(X)
        Z = p[:-1].transform(X)
        linhas.append({"k": k, "inercia": p["kmeans"].inertia_,
                       "silhueta": silhouette_score(Z, p["kmeans"].labels_, sample_size=3000,
                                                    random_state=RANDOM_STATE)})
    return pd.DataFrame(linhas).set_index("k")


def perfil_clusters(df: pd.DataFrame, rotulo: str = "cluster") -> pd.DataFrame:
    cols = list(VARS_CLUSTER)
    perfil = df.groupby(rotulo)[cols].median()
    perfil.insert(0, "municípios", df.groupby(rotulo).size())
    return perfil


def nomear_clusters(perfil: pd.DataFrame) -> dict:
    """Atribui nomes interpretáveis a partir do perfil mediano de cada grupo.

    Regras simples e transparentes (aplicadas em ordem), independentes do número do cluster –
    o que mantém os nomes estáveis mesmo que o K-Means numere os grupos de outra forma.
    """
    nomes = {}
    for c, p in perfil.iterrows():
        if p["taxa_2024"] < 50:
            nome = "Vulnerabilidade social e educacional"
        elif p["variacao_taxa"] <= -5:
            nome = "Queda recente de desempenho"
        elif p["taxa_2024"] >= 70 and p["idhm"] < 0.65:
            nome = "Supera o contexto"
        elif p["taxa_2024"] >= 70:
            nome = "Bom desempenho e contexto favorável"
        else:
            nome = "Desempenho intermediário" + (" (centros maiores)" if p["log_populacao"] >= 4.5 else "")
        nomes[c] = nome
    vistos = {}
    for c, n in nomes.items():  # garante nomes únicos
        vistos[n] = vistos.get(n, 0) + 1
        if vistos[n] > 1:
            nomes[c] = f"{n} ({vistos[n]})"
    return nomes


def prioridade(taxa: pd.Series, risco: pd.Series, corte_taxa: float = 50, corte_risco: float = 0.5) -> pd.Series:
    """Matriz de prioridade: combina nível atual (taxa) e risco de não atingir a próxima meta."""
    return pd.Series(np.select(
        [(taxa < corte_taxa) & (risco >= corte_risco), (taxa < corte_taxa), (risco >= corte_risco)],
        ["1 – Crítica", "2 – Alta (baixo nível)", "3 – Atenção (risco de meta)"],
        "4 – Monitoramento"), index=taxa.index)


# ---------------------------------------------------------------------------------------------
# Projeção robusta: regressão da taxa do ano seguinte + probabilidade de ficar abaixo da meta
# ---------------------------------------------------------------------------------------------
# A meta não entra como variável: só é usada na comparação final. Assim o modelo funciona para
# qualquer meta futura sem extrapolar (o esforço exigido em 2025 vai muito além do visto em 2024).
NUM_PROJECAO = [c for c in NUM_MUNICIPIO if c not in ("meta_alvo", "esforco_meta")]
CAT_PROJECAO = CAT_MUNICIPIO


class RiscoMetaRegressao:
    """Prevê a taxa de alfabetização do ano seguinte e converte em probabilidade de não atingir a meta.

    risco = P(taxa_{t+1} < meta_{t+1}) = Φ((meta − taxa_prevista) / σ_porte)

    σ_porte é o desvio padrão dos resíduos fora da amostra (validação cruzada) por porte do município –
    municípios pequenos têm poucas crianças avaliadas e, portanto, taxas mais voláteis.
    """

    def __init__(self, pipeline, cv=None):
        self.pipeline = pipeline
        self.cv = cv

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.model_selection import KFold, cross_val_predict
        cv = self.cv or KFold(5, shuffle=True, random_state=RANDOM_STATE)
        oof = cross_val_predict(self.pipeline, X, y, cv=cv)
        residuos = y - oof
        self.sigma_porte_ = residuos.groupby(X["porte_municipio"]).std()
        self.sigma_global_ = residuos.std()
        self.oof_ = pd.Series(oof, index=X.index)
        self.pipeline.fit(X, y)
        return self

    def prever_taxa(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.pipeline.predict(X), 0, 100)

    def sigma(self, X: pd.DataFrame) -> np.ndarray:
        return X["porte_municipio"].map(self.sigma_porte_).fillna(self.sigma_global_).to_numpy()

    def risco(self, X: pd.DataFrame, meta) -> np.ndarray:
        from scipy.stats import norm
        return norm.cdf((np.asarray(meta, dtype=float) - self.prever_taxa(X)) / self.sigma(X))
