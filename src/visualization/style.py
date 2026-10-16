"""Estilo visual único para todos os gráficos do projeto (matplotlib)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.config import IMAGES_DIR

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AZUL, LARANJA = SERIES[0], SERIES[1]
CINZA = "#898781"
TEXTO, TEXTO_2 = "#0b0b0b", "#52514e"
GRADE, EIXO, SUPERFICIE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Status (reservados para atingiu / não atingiu meta; sempre acompanhados de rótulo)
STATUS_OK, STATUS_CRITICO, STATUS_ALERTA = "#0ca30c", "#d03b3b", "#fab219"

SEQ_AZUL = LinearSegmentedColormap.from_list(
    "seq_azul", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGENTE = LinearSegmentedColormap.from_list(
    "div_azul_vermelho", ["#184f95", "#6da7ec", "#f0efec", "#ec7b7a", "#a32c2c"])


def aplicar_estilo() -> None:
    """Aplica o padrão visual do projeto a todos os gráficos matplotlib da sessão."""
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 10,
        "text.color": TEXTO, "axes.labelcolor": TEXTO_2, "axes.titlecolor": TEXTO,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": EIXO, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": CINZA, "ytick.color": CINZA, "xtick.labelcolor": TEXTO_2, "ytick.labelcolor": TEXTO_2,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9,
        "patch.linewidth": 0,
    })


def salvar(fig, nome: str) -> None:
    """Salva a figura em images/ (usada no README e nos relatórios)."""
    fig.savefig(IMAGES_DIR / f"{nome}.png")
