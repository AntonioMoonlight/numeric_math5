from typing import Callable, Dict, Optional, Sequence
import matplotlib.pyplot as plt


def build_interpolation_plot(
    x_nodes: Sequence[float],
    y_nodes: Sequence[float],
    dense_curves: Dict[str, Sequence[float]],
    x_dense: Sequence[float],
    x_target: float,
    target_results: Dict[str, float],
    true_func: Optional[Callable[[float], float]] = None,
) -> plt.Figure:
    """Строит график со сплошными линиями и фиксированной внешней легендой."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=120)

    # Исходная функция
    if true_func is not None:
        y_true = [true_func(xi) for xi in x_dense]
        ax.plot(
            x_dense,
            y_true,
            label="f(x)",
            color="#495057",
            linestyle="-",
            lw=1.5,
            alpha=0.75,
        )

    palette = {
        "Лагранж": {"color": "#1f77b4", "lw": 2.0},
        "Ньютон": {"color": "#e65100", "lw": 1.8},
        "Гаусс": {"color": "#2e7d32", "lw": 1.8},
        "Стирлинг": {"color": "#9467bd", "lw": 1.8},
        "Бессель": {"color": "#8c564b", "lw": 1.8},
    }

    for name, y_vals in dense_curves.items():
        base_key = next((k for k in ("Ньютон", "Гаусс", "Стирлинг", "Бессель", "Лагранж") if k in name), "Лагранж")
        style = palette.get(base_key, {"color": "gray", "lw": 1.5})
        ax.plot(
            x_dense,
            y_vals,
            label=name,
            color=style["color"],
            linestyle="-",
            lw=style["lw"],
        )

    ax.scatter(
        x_nodes,
        y_nodes,
        color="#212529",
        s=32,
        zorder=5,
        label="Узлы",
    )

    ax.axvline(
        x=x_target,
        color="#868e96",
        linestyle="-",
        lw=1.0,
        alpha=0.7,
        label=f"x* = {x_target:.4f}",
    )
    for name, val in target_results.items():
        if name in dense_curves:
            base_key = next((k for k in ("Ньютон", "Гаусс", "Стирлинг", "Бессель", "Лагранж") if k in name), "Лагранж")
            color = palette.get(base_key, {"color": "black"})["color"]
            ax.scatter(x_target, val, color=color, s=65, marker="x", zorder=6)

    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.grid(True, linestyle="-", alpha=0.3)

    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8.5,
        frameon=True,
    )
    fig.tight_layout()
    return fig