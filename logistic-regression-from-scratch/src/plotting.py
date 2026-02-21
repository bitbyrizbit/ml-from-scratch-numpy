import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def handle_plot(
    filename: str,
    save_plots: bool,
    plots_dir: str = "outputs/plots"
):
    full_dir = PROJECT_ROOT / plots_dir

    if save_plots:
        full_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(full_dir / filename, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
