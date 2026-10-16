"""Métricas e gráficos de avaliação de modelos de classificação binária."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.visualization.style import CINZA, SERIES, TEXTO_2

SCORING_CV = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "log_loss": "neg_log_loss",
    "brier": "neg_brier_score",
    "balanced_acc": "balanced_accuracy",
}


def metricas(y_true, proba, limiar: float = 0.5, classe_foco: int = 1, nome_classe: str | None = None) -> dict:
    """Métricas de classificação binária.

    ``proba`` é sempre a probabilidade da classe 1. Precisão, Recall, F1 e PR-AUC são calculados para a
    ``classe_foco`` (a classe de interesse do negócio). Com ``nome_classe``, essas colunas recebem o nome da
    classe entre parênteses, para deixar claro a qual classe se referem.
    Ex.: ``metricas(y, p, classe_foco=0, nome_classe="não alfabetizado")``.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = (proba >= limiar).astype(int)
    y_foco = (y_true == classe_foco).astype(int)
    pred_foco = (pred == classe_foco).astype(int)
    score_foco = proba if classe_foco == 1 else 1 - proba
    sufixo = f" ({nome_classe})" if nome_classe else ""
    return {
        "ROC-AUC": roc_auc_score(y_true, proba),
        f"PR-AUC{sufixo}": average_precision_score(y_foco, score_foco),
        "Log loss": log_loss(y_true, proba),
        "Brier": brier_score_loss(y_true, proba),
        "Acurácia balanceada": balanced_accuracy_score(y_true, pred),
        f"Precisão{sufixo}": precision_score(y_foco, pred_foco, zero_division=0),
        f"Recall{sufixo}": recall_score(y_foco, pred_foco),
        f"F1{sufixo}": f1_score(y_foco, pred_foco),
        "Limiar": limiar,
    }


def resumo_cv(resultados: dict) -> pd.DataFrame:
    """Transforma a saída de cross_validate (vários modelos) em tabela média ± desvio."""
    linhas = []
    for nome, r in resultados.items():
        linha = {"modelo": nome}
        for k in SCORING_CV:
            v = r[f"test_{k}"]
            v = -v if k in ("log_loss", "brier") else v
            linha[k] = f"{np.mean(v):.4f} ± {np.std(v):.4f}"
            linha[f"_{k}"] = np.mean(v)
        if "train_roc_auc" in r:
            linha["roc_auc_treino"] = np.mean(r["train_roc_auc"])
            linha["gap_overfitting"] = np.mean(r["train_roc_auc"]) - np.mean(r["test_roc_auc"])
        linha["tempo_ajuste_s"] = np.mean(r["fit_time"])
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("modelo")


def limiar_otimo_f1(y_true, proba, alvo_classe: int = 0) -> float:
    """Limiar que maximiza o F1 da classe de interesse (padrão: NÃO alfabetizado = 0)."""
    if alvo_classe == 0:
        p, r, t = precision_recall_curve(1 - np.asarray(y_true), 1 - proba)
        f1 = 2 * p * r / (p + r + 1e-12)
        return float(1 - t[np.argmax(f1[:-1])])
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.argmax(f1[:-1])])


def grafico_avaliacao(y_true, probas: dict, titulo: str = ""):
    """ROC, Precision-Recall e calibração para um ou mais modelos."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for (nome, p), cor in zip(probas.items(), SERIES):
        fpr, tpr, _ = roc_curve(y_true, p)
        axes[0].plot(fpr, tpr, color=cor, label=f"{nome} (AUC {roc_auc_score(y_true, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_true, p)
        axes[1].plot(rc, pr, color=cor, label=f"{nome} (AP {average_precision_score(y_true, p):.3f})")
        fr, mp = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        axes[2].plot(mp, fr, marker="o", color=cor, label=nome)
    axes[0].plot([0, 1], [0, 1], color=CINZA, linestyle="--", linewidth=1)
    axes[0].set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos", title="Curva ROC")
    axes[1].axhline(np.mean(y_true), color=CINZA, linestyle="--", linewidth=1)
    axes[1].set(xlabel="Recall", ylabel="Precisão", title="Curva Precision-Recall (classe 1)")
    axes[2].plot([0, 1], [0, 1], color=CINZA, linestyle="--", linewidth=1)
    axes[2].set(xlabel="Probabilidade prevista", ylabel="Frequência observada", title="Calibração")
    for ax in axes:
        ax.legend(loc="lower right", fontsize=8)
    if titulo:
        fig.suptitle(titulo, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def grafico_matriz_confusao(y_true, pred, rotulos=("Não alfabetizado", "Alfabetizado"), titulo=""):
    """Matriz de confusão com contagens e percentual por classe real (linha)."""
    cm = confusion_matrix(y_true, pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.0%})", ha="center", va="center",
                    color="white" if cm_pct[i, j] > 0.5 else TEXTO_2, fontsize=10)
    ax.set_xticks([0, 1], [f"Prev.: {r}" for r in rotulos], fontsize=8)
    ax.set_yticks([0, 1], [f"Real: {r}" for r in rotulos], fontsize=8)
    ax.grid(False)
    ax.set_title(titulo or "Matriz de confusão")
    return fig
