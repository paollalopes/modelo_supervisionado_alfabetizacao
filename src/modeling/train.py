"""Treinamento, validação e otimização dos modelos."""

from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import (GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold,
                                     cross_validate)

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.evaluation.metrics import SCORING_CV


def dividir_por_grupo(X, y, grupos, test_size: float = 0.2, seed: int = RANDOM_STATE):
    """Hold-out por município: nenhum município aparece ao mesmo tempo em treino e teste."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_tr, idx_te = next(gss.split(X, y, grupos))
    return idx_tr, idx_te


def cv_por_grupo(n_splits: int = 5, seed: int = RANDOM_STATE):
    """K-Fold estratificado (mantém a proporção do alvo) e agrupado por município."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def avaliar_cv(pipeline, X, y, grupos=None, cv=None, n_jobs: int = 1) -> dict:
    """Validação cruzada com as métricas de ``SCORING_CV`` (treino e validação). Por padrão, agrupa por município."""
    cv = cv or cv_por_grupo()
    return cross_validate(pipeline, X, y, groups=grupos, cv=cv, scoring=SCORING_CV,
                          return_train_score=True, n_jobs=n_jobs)


ESPACO_LGBM = {
    "modelo__n_estimators": randint(150, 800),
    "modelo__learning_rate": loguniform(0.01, 0.1),
    "modelo__num_leaves": randint(4, 40),
    "modelo__min_child_samples": randint(50, 600),
    "modelo__subsample": uniform(0.6, 0.4),
    "modelo__colsample_bytree": uniform(0.4, 0.6),
    "modelo__reg_lambda": loguniform(0.1, 30),
    "modelo__reg_alpha": loguniform(1e-3, 10),
}

ESPACO_LOGISTICA = {"modelo__C": loguniform(1e-3, 10)}


def otimizar(pipeline, espaco: dict, X, y, grupos, n_iter: int = 20, scoring: str = "roc_auc",
             n_jobs: int = -1, seed: int = RANDOM_STATE) -> RandomizedSearchCV:
    """Busca aleatória de hiperparâmetros com validação cruzada agrupada por município."""
    # Sem grupos (ex.: 1 linha por município), usa K-Fold estratificado simples
    cv = cv_por_grupo(seed=seed) if grupos is not None else StratifiedKFold(5, shuffle=True, random_state=seed)
    busca = RandomizedSearchCV(pipeline, espaco, n_iter=n_iter, cv=cv,
                               scoring=scoring, n_jobs=n_jobs, random_state=seed,
                               return_train_score=True, refit=True)
    busca.fit(X, y, groups=grupos)
    return busca


def bootstrap_metrica(y_true, proba, metrica, n: int = 500, seed: int = RANDOM_STATE, grupos=None):
    """Intervalo de confiança (95%) por bootstrap. Se ``grupos`` for informado, reamostra municípios
    inteiros (bootstrap em cluster), respeitando a dependência entre alunos do mesmo município."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    valores = []
    if grupos is None:
        for _ in range(n):
            i = rng.integers(0, len(y_true), len(y_true))
            valores.append(metrica(y_true[i], proba[i]))
    else:
        grupos = np.asarray(grupos)
        unicos = np.unique(grupos)
        idx_por_grupo = {gr: np.where(grupos == gr)[0] for gr in unicos}
        for _ in range(n):
            sorteio = rng.choice(unicos, len(unicos), replace=True)
            i = np.concatenate([idx_por_grupo[gr] for gr in sorteio])
            valores.append(metrica(y_true[i], proba[i]))
    return np.percentile(valores, [2.5, 50, 97.5])


def _versoes_bibliotecas() -> dict:
    """Versões das bibliotecas necessárias para recarregar o modelo de forma idêntica."""
    versoes = {"python": platform.python_version()}
    for pacote in ["scikit-learn", "lightgbm", "xgboost", "pandas", "numpy"]:
        try:
            versoes[pacote] = version(pacote)
        except PackageNotFoundError:
            versoes[pacote] = None
    return versoes


def salvar_modelo(modelo, nome: str, metadados: dict | None = None) -> None:
    """Salva o modelo em models/ e os metadados (incluindo versões das bibliotecas) em reports/."""
    joblib.dump(modelo, MODELS_DIR / f"{nome}.joblib")
    metadados = {**(metadados or {}), "versoes": _versoes_bibliotecas()}
    with open(REPORTS_DIR / f"{nome}_metadados.json", "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2, default=str)


def carregar_modelo(nome: str):
    """Carrega um modelo salvo em models/. Use as mesmas versões registradas nos metadados."""
    return joblib.load(MODELS_DIR / f"{nome}.joblib")


def tabela_busca(busca: RandomizedSearchCV, top: int = 10) -> pd.DataFrame:
    """As ``top`` melhores combinações da busca, com a diferença (gap) entre treino e validação."""
    r = pd.DataFrame(busca.cv_results_)
    cols = [c for c in r.columns if c.startswith("param_")]
    r = r[cols + ["mean_test_score", "std_test_score", "mean_train_score"]]
    r.columns = [c.replace("param_modelo__", "") for c in r.columns]
    r["gap"] = r["mean_train_score"] - r["mean_test_score"]
    return r.sort_values("mean_test_score", ascending=False).head(top)
