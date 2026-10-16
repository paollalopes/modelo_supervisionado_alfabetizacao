"""Pipelines de Machine Learning (scikit-learn) com pré-processamento integrado ao modelo.

Toda transformação que "aprende" com os dados (medianas da imputação, médias/desvios da padronização, categorias do One-Hot) é ajustada DENTRO da pipeline. 
Assim, na validação cruzada, cada fold ajusta o pré-processamento apenas com os dados de treino daquele fold – prevenindo data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import RANDOM_STATE

# --------------------------------------------------------------------------------------
# Conjuntos de variáveis – modelo de ALUNOS
# --------------------------------------------------------------------------------------
TERRITORIO_CAT = ["sigla_uf", "regiao", "porte_municipio"]
TERRITORIO_NUM = ["capital", "latitude", "longitude", "log_populacao", "crescimento_pop_2010_2022",
                  "taxa_urbanizacao_2010"]
SOCIOECONOMICAS = ["idhm", "idhm_educacao", "idhm_longevidade", "idhm_renda", "renda_per_capita", "gini",
                   "pct_extrema_pobreza", "pct_pobres", "pct_criancas_extrema_pobreza",
                   "taxa_analfabetismo_18mais", "anos_estudo_esperados", "taxa_freq_escola_4a5",
                   "taxa_freq_escola_5a6", "taxa_atraso_2anos_fund", "pct_criancas_lar_sem_fund_completo",
                   "pct_agua_encanada", "pct_sem_agua_esgoto_adequado", "pct_energia_eletrica",
                   "mortalidade_ate_5anos", "esperanca_vida", "razao_dependencia",
                   "grau_formalizacao_trabalho", "taxa_desocupacao"]
HISTORICO_EDUCACIONAL = ["taxa_alfabetizacao_estadual_ant", "taxa_alfabetizacao_municipal_ant",
                         "taxa_alfabetizacao_publica_ant", "media_portugues_estadual_ant",
                         "media_portugues_municipal_ant", "media_portugues_publica_ant",
                         "participacao_ant", "nivel_alfabetizacao_ant", "taxa_uf_publica_ant",
                         "taxa_rede_aluno_ant", "media_rede_aluno_ant"]
METAS = ["meta_alfabetizacao_2024", "meta_alfabetizacao_2025", "meta_alfabetizacao_2026",
         "meta_alfabetizacao_2027", "meta_alfabetizacao_2028", "meta_alfabetizacao_2029",
         "esforco_meta_2024"]
ALUNO_CAT = ["rede_ensino"]

CATEGORICAS = ALUNO_CAT + TERRITORIO_CAT
NUMERICAS = TERRITORIO_NUM + SOCIOECONOMICAS + HISTORICO_EDUCACIONAL + METAS
ALVO = "alfabetizado"
GRUPO = "id_municipio"


# --------------------------------------------------------------------------------------
# Engenharia de atributos (dentro da pipeline)
# --------------------------------------------------------------------------------------
class AtributosDerivados(BaseEstimator, TransformerMixin):
    """Cria variáveis derivadas. Não aprende nada com os dados (sem risco de leakage),
    mas fica dentro da pipeline para que treino e produção apliquem exatamente a mesma lógica.

    ATENÇÃO: ao incluir ou remover um atributo aqui, atualize também ``novas_colunas()``.
    As duas precisam listar as mesmas colunas, senão o pré-processador ignora a coluna nova."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if {"taxa_alfabetizacao_estadual_ant", "taxa_alfabetizacao_municipal_ant"} <= set(X.columns):
            X["tem_rede_estadual"] = X["taxa_alfabetizacao_estadual_ant"].notna().astype(float)
            X["dif_estadual_municipal_ant"] = (X["taxa_alfabetizacao_estadual_ant"]
                                               - X["taxa_alfabetizacao_municipal_ant"])
        if {"taxa_rede_aluno_ant", "taxa_uf_publica_ant"} <= set(X.columns):
            X["gap_rede_aluno_vs_uf_ant"] = X["taxa_rede_aluno_ant"] - X["taxa_uf_publica_ant"]
        if {"renda_per_capita"} <= set(X.columns):
            X["log_renda_per_capita"] = np.log1p(X["renda_per_capita"])
        return X


def novas_colunas(numericas: list[str]) -> list[str]:
    """Retorna as colunas derivadas que serão criadas a partir das numéricas disponíveis.
    Precisa espelhar a lógica de ``AtributosDerivados.transform``."""
    s = set(numericas)
    out = []
    if {"taxa_alfabetizacao_estadual_ant", "taxa_alfabetizacao_municipal_ant"} <= s:
        out += ["tem_rede_estadual", "dif_estadual_municipal_ant"]
    if {"taxa_rede_aluno_ant", "taxa_uf_publica_ant"} <= s:
        out += ["gap_rede_aluno_vs_uf_ant"]
    if "renda_per_capita" in s:
        out += ["log_renda_per_capita"]
    return out


# --------------------------------------------------------------------------------------
# Pré-processamento
# --------------------------------------------------------------------------------------
def criar_preprocessador(numericas: list[str], categoricas: list[str], escalonar: bool = True):
    """ColumnTransformer: imputação + transformação de numéricas e categóricas."""
    passos_num = [("imputacao", SimpleImputer(strategy="median", add_indicator=True))]
    if escalonar:  # necessário para modelos lineares; indiferente para árvores
        passos_num.append(("padronizacao", StandardScaler()))
    num = Pipeline(passos_num)
    cat = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num, numericas + novas_colunas(numericas)),
        ("cat", cat, categoricas),
    ], verbose_feature_names_out=False)


def modelos_candidatos() -> dict:
    """Modelos comparados. Hiperparâmetros iniciais conservadores (anti-overfitting)."""
    return {
        "Regressão Logística": LogisticRegression(C=1.0, max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features="sqrt",
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                               min_samples_leaf=100, l2_regularization=1.0,
                                                               random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
                                 eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   reg_lambda=1.0, verbose=-1, n_jobs=-1, random_state=RANDOM_STATE),
    }


def criar_pipeline(modelo, numericas: list[str] | None = None, categoricas: list[str] | None = None) -> Pipeline:
    """Pipeline completa: atributos derivados → pré-processamento → modelo.
    A padronização só é aplicada à Regressão Logística. Sem listas, usa as variáveis do modelo de alunos."""
    numericas = NUMERICAS if numericas is None else numericas
    categoricas = CATEGORICAS if categoricas is None else categoricas
    escalonar = isinstance(modelo, LogisticRegression)
    return Pipeline([
        ("atributos", AtributosDerivados()),
        ("preprocessamento", criar_preprocessador(numericas, categoricas, escalonar)),
        ("modelo", modelo),
    ])


def selecionar_X(df: pd.DataFrame, numericas: list[str] | None = None,
                 categoricas: list[str] | None = None) -> pd.DataFrame:
    """Seleciona do DataFrame apenas as colunas usadas pelo modelo (por padrão, as do modelo de alunos)."""
    numericas = NUMERICAS if numericas is None else numericas
    categoricas = CATEGORICAS if categoricas is None else categoricas
    return df[numericas + categoricas]
