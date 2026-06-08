from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
NUNITO_FONT = ROOT / "assets" / "fonts" / "Nunito-VariableFont_wght.ttf"

COLORS = {
    "ink": "#17212B",
    "muted": "#68727D",
    "grid": "#D9E0E7",
    "panel": "#F7F9FB",
    "prosst": "#5B3F99",
    "esm2": "#1F7A8C",
    "evodiff": "#C4472D",
    "fit": "#E2A93B",
}


def register_nunito() -> str:
    if NUNITO_FONT.exists():
        font_manager.fontManager.addfont(str(NUNITO_FONT))
        return "Nunito"
    return "DejaVu Sans"


def set_prospero_style() -> None:
    family = register_nunito()
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.weight": "light",
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "axes.labelweight": "light",
            "axes.edgecolor": COLORS["grid"],
            "axes.linewidth": 0.9,
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "legend.fontsize": 11,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
