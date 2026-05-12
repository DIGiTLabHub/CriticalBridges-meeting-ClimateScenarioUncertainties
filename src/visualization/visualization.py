from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


DEFAULT_RECORDER_DATA_DIR = Path("RecorderData")
DEFAULT_YIELD_RESULTS = DEFAULT_RECORDER_DATA_DIR / "Yield_Results_by_Scenario.xlsx"
DEFAULT_RESULTS_DIR = DEFAULT_RECORDER_DATA_DIR / "results" / "Tuple_Data_Process"
DEFAULT_GBR_DIR = DEFAULT_RESULTS_DIR / "ML_Surrogate_Credal"
DEFAULT_SVR_DIR = DEFAULT_RESULTS_DIR / "ML_Surrogate_Credal_SVR"
DEFAULT_OUTPUT_DIR = DEFAULT_RECORDER_DATA_DIR / "results" / "visualizations"

CAPACITY_COLUMNS = {
    "Vy_kN": "Yield Base Shear (kN)",
    "dy_mm": "Yield Displacement (mm)",
    "My_kNm": "Yield Moment (kNm)",
    "Thy_rad": "Yield Rotation (rad)",
}


def _import_plot_deps():
    import pandas as pd
    import matplotlib.pyplot as plt

    return pd, plt


def load_yield_results(workbook_path: Path = DEFAULT_YIELD_RESULTS):
    pd, _ = _import_plot_deps()
    if not workbook_path.exists():
        raise FileNotFoundError(f"Yield results workbook not found: {workbook_path}")
    return pd.read_excel(workbook_path, sheet_name=None)


def plot_capacity_vs_scour(
    workbook_path: Path = DEFAULT_YIELD_RESULTS,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metrics: Iterable[str] = CAPACITY_COLUMNS.keys(),
) -> list[Path]:
    pd, plt = _import_plot_deps()
    sheets = load_yield_results(workbook_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    for metric in metrics:
        label = CAPACITY_COLUMNS[metric]
        fig, ax = plt.subplots(figsize=(8, 5))

        for scenario, df in sheets.items():
            required = {"Scour_Depth_mm", metric}
            if not required.issubset(df.columns):
                continue

            plot_df = (
                df[["Scour_Depth_mm", metric]].dropna().sort_values("Scour_Depth_mm")
            )
            if plot_df.empty:
                continue

            ax.plot(
                plot_df["Scour_Depth_mm"],
                plot_df[metric],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=scenario,
            )

        ax.set_xlabel("Scour Depth (mm)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Scour Depth")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()

        output_path = output_dir / f"capacity_vs_scour_{metric}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        saved_files.append(output_path)

    return saved_files


def plot_credal_bounds(
    model_dir: Path,
    *,
    output_dir: Path,
    model_label: str,
) -> list[Path]:
    pd, plt = _import_plot_deps()
    if not model_dir.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    for bounds_file in sorted(model_dir.glob("Credal_Bounds_*.xlsx")):
        target = bounds_file.stem.replace("Credal_Bounds_", "")
        df = pd.read_excel(bounds_file)
        required = {"Actual", "LowerBound", "UpperBound", "MedianPrediction"}
        if not required.issubset(df.columns):
            continue

        plot_df = df.sort_values("Actual").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(
            plot_df["Actual"], plot_df["MedianPrediction"], label="Median Prediction"
        )
        ax.fill_between(
            plot_df["Actual"],
            plot_df["LowerBound"],
            plot_df["UpperBound"],
            alpha=0.25,
            label="Credal Bounds",
        )
        diag_min = min(plot_df["Actual"].min(), plot_df["MedianPrediction"].min())
        diag_max = max(plot_df["Actual"].max(), plot_df["MedianPrediction"].max())
        ax.plot(
            [diag_min, diag_max], [diag_min, diag_max], "k--", linewidth=1, label="1:1"
        )
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{model_label} Credal Bounds: {target}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()

        output_path = output_dir / f"{model_label.lower()}_credal_{target}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        saved_files.append(output_path)

    return saved_files


def plot_model_performance(
    summary_excel: Path,
    *,
    output_dir: Path,
    model_label: str,
) -> list[Path]:
    pd, plt = _import_plot_deps()
    if not summary_excel.exists():
        return []

    df = pd.read_excel(summary_excel)
    required = {"Target", "R2", "RMSE"}
    if not required.issubset(df.columns):
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    for metric in ("R2", "RMSE"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(df["Target"], df[metric], color="#4C78A8")
        ax.set_title(f"{model_label} {metric} by Target")
        ax.set_xlabel("Target")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        output_path = output_dir / f"{model_label.lower()}_{metric.lower()}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        saved_files.append(output_path)

    return saved_files


def create_all_plots(
    *,
    yield_results: Path = DEFAULT_YIELD_RESULTS,
    gbr_dir: Path = DEFAULT_GBR_DIR,
    svr_dir: Path = DEFAULT_SVR_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    saved_files: list[Path] = []
    saved_files.extend(
        plot_capacity_vs_scour(yield_results, output_dir=output_dir / "capacity")
    )
    saved_files.extend(
        plot_credal_bounds(gbr_dir, output_dir=output_dir / "gbr", model_label="GBR")
    )
    saved_files.extend(
        plot_credal_bounds(svr_dir, output_dir=output_dir / "svr", model_label="SVR")
    )
    saved_files.extend(
        plot_model_performance(
            gbr_dir.parent / "Credal_Model_Performance_GBR_Filtered_AllTuples.xlsx",
            output_dir=output_dir / "gbr",
            model_label="GBR",
        )
    )
    saved_files.extend(
        plot_model_performance(
            svr_dir.parent / "Credal_Model_Performance_SVR_Filtered_AllTuples.xlsx",
            output_dir=output_dir / "svr",
            model_label="SVR",
        )
    )
    return saved_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create supported visualization outputs from postprocessing and surrogate-training artifacts."
    )
    parser.add_argument("--yield-results", type=Path, default=DEFAULT_YIELD_RESULTS)
    parser.add_argument("--gbr-dir", type=Path, default=DEFAULT_GBR_DIR)
    parser.add_argument("--svr-dir", type=Path, default=DEFAULT_SVR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    saved = create_all_plots(
        yield_results=args.yield_results,
        gbr_dir=args.gbr_dir,
        svr_dir=args.svr_dir,
        output_dir=args.output_dir,
    )
    if saved:
        print("✅ Saved plots:")
        for path in saved:
            print(f"  - {path}")
    else:
        print(
            "⚠️ No plots were generated. Check that the expected input artifacts exist."
        )


if __name__ == "__main__":
    main()
