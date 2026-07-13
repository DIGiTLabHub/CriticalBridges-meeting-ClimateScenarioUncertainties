#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from src.postprocessing.bilinear_fit import fit_bilinear_profile
except Exception:
    try:
        from .bilinear_fit import fit_bilinear_profile  # type: ignore
    except Exception:
        from bilinear_fit import fit_bilinear_profile  # type: ignore


DEFAULT_SCENARIOS = ("missouri", "colorado", "extreme")
DEFAULT_ELEMENT_IDS = (3101, 3201, 3301)
FORCE_COLUMNS = [
    "time",
    "P_i",
    "V2_i",
    "V3_i",
    "T_i",
    "M2_i",
    "M3_i",
    "P_j",
    "V2_j",
    "V3_j",
    "T_j",
    "M2_j",
    "M3_j",
]


@dataclass(frozen=True)
class RecorderFilesConfig:
    displacement_file: str = "Displacement.5201.out"
    column_displacement_file: str = "ColDisplacement.3201.out"
    force_file_template: str = "ColLocForce.{element_id}.out"
    moment_element_id: int = 3201


def parse_scour_depth_mm(scour_folder_name: str) -> float:
    return float(scour_folder_name.split("_", 1)[1])


def load_recorder_folder_data(
    folder: Path,
    *,
    element_ids: Iterable[int] = DEFAULT_ELEMENT_IDS,
    files: RecorderFilesConfig = RecorderFilesConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    disp_file = folder / files.displacement_file
    col_disp_file = folder / files.column_displacement_file
    moment_file = folder / files.force_file_template.format(
        element_id=files.moment_element_id
    )

    df_disp = pd.read_csv(disp_file, sep=r"\s+", header=None)
    df_disp.columns = ["time", "ux", "uy", "uz", "rx", "ry", "rz"]
    disp_y = df_disp["uy"].abs()

    df_col = pd.read_csv(col_disp_file, sep=r"\s+", header=None)
    theta_y = df_col.iloc[:, 10].abs()

    base_shear = None
    for element_id in element_ids:
        force_file = folder / files.force_file_template.format(element_id=element_id)
        df_force = pd.read_csv(force_file, sep=r"\s+", header=None, names=FORCE_COLUMNS)
        shear = df_force["V2_j"]
        base_shear = shear if base_shear is None else base_shear + shear

    if base_shear is None:
        raise ValueError(f"No base shear data found in folder: {folder}")

    df_moment = pd.read_csv(moment_file, sep=r"\s+", header=None, names=FORCE_COLUMNS)
    moment = df_moment["M3_j"].abs()

    min_len = min(len(disp_y), len(base_shear), len(moment), len(theta_y))
    # Recorder displacements are already in mm because the model uses N-mm-MPa units.
    d = disp_y[:min_len].to_numpy()
    f = base_shear[:min_len].abs().to_numpy() / 1000.0
    m = moment[:min_len].to_numpy() / 1e6
    t = theta_y[:min_len].to_numpy()
    return d, f, m, t


def compute_yield_quantities(
    d: np.ndarray,
    f: np.ndarray,
    m: np.ndarray,
    t: np.ndarray,
    *,
    num_grid: int = 200,
) -> dict[str, float]:
    k1, k2, dy, vy = fit_bilinear_profile(d, f, num_grid=num_grid)
    if k1 is None or k2 is None or dy is None or vy is None:
        raise ValueError("Bilinear fit failed to produce valid parameters.")

    k1_value = float(k1)
    k2_value = float(k2)
    dy_value = float(dy)
    vy_value = float(vy)

    i_yield = int(np.argmin(np.abs(d - dy_value)))
    my = float(m[i_yield])
    thy = float(t[i_yield])

    return {
        "dy_mm": dy_value,
        "Vy_kN": vy_value,
        "My_kNm": my,
        "Thy_rad": thy,
        "k1_kN_per_mm": k1_value,
        "k2_kN_per_mm": k2_value,
    }


def _save_shear_plot(
    *,
    d: np.ndarray,
    f: np.ndarray,
    dy: float,
    vy: float,
    k1: float,
    k2: float,
    scenario: str,
    scour_depth_mm: float,
    output_file: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Plotting requested but matplotlib is not installed. "
            "Install matplotlib or run with --no-plot."
        ) from exc

    d_range = np.linspace(d.min(), d.max(), 500)
    f_model = np.where(d_range < dy, k1 * d_range, k2 * d_range + (k1 - k2) * dy)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d, f, color="gray", alpha=0.5)
    ax.plot(d_range, f_model, color="blue")
    ax.scatter([dy], [vy], color="red", zorder=10)

    props = dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8)
    textstr = "\n".join(
        (
            r"Pushover: gray line",
            r"Bilinear Fit: blue line",
            r"Yield: red dot",
            "",
            rf"$k_1 = {k1:.2f}\ \mathrm{{kN/mm}}$",
            rf"$k_2 = {k2:.2f}\ \mathrm{{kN/mm}}$",
            rf"$d_y = {dy:.2f}\ \mathrm{{mm}}$",
            rf"$V_y = {vy:.2f}\ \mathrm{{kN}}$",
        )
    )
    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    ax.set_title(f"{scenario} | Scour = {scour_depth_mm} mm")
    ax.set_xlabel("Displacement $d$ (mm)")
    ax.set_ylabel("Base Shear $V_y$ (kN)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file)
    plt.close(fig)


def process_scour_folder(
    folder: Path,
    *,
    scenario: str,
    element_ids: Iterable[int] = DEFAULT_ELEMENT_IDS,
    files: RecorderFilesConfig = RecorderFilesConfig(),
    num_grid: int = 200,
    save_plot: bool = False,
    plot_output_dir: Path | None = None,
) -> dict[str, float]:
    scour_depth = parse_scour_depth_mm(folder.name)
    d, f, m, t = load_recorder_folder_data(folder, element_ids=element_ids, files=files)
    yield_data = compute_yield_quantities(d, f, m, t, num_grid=num_grid)

    row = {
        "Scour_Depth_mm": scour_depth,
        "dy_mm": round(yield_data["dy_mm"], 3),
        "Vy_kN": round(yield_data["Vy_kN"], 2),
        "My_kNm": round(yield_data["My_kNm"], 2),
        "Thy_rad": round(yield_data["Thy_rad"], 5),
        "k1_kN_per_mm": round(yield_data["k1_kN_per_mm"], 2),
        "k2_kN_per_mm": round(yield_data["k2_kN_per_mm"], 2),
    }

    if save_plot:
        if plot_output_dir is None:
            raise ValueError("plot_output_dir must be provided when save_plot=True")
        plot_file = plot_output_dir / scenario / f"{folder.name}.png"
        _save_shear_plot(
            d=d,
            f=f,
            dy=yield_data["dy_mm"],
            vy=yield_data["Vy_kN"],
            k1=yield_data["k1_kN_per_mm"],
            k2=yield_data["k2_kN_per_mm"],
            scenario=scenario,
            scour_depth_mm=scour_depth,
            output_file=plot_file,
        )

    return row


def batch_process_scenarios(
    recorder_data_dir: Path,
    *,
    scenarios: Iterable[str] = DEFAULT_SCENARIOS,
    element_ids: Iterable[int] = DEFAULT_ELEMENT_IDS,
    files: RecorderFilesConfig = RecorderFilesConfig(),
    num_grid: int = 200,
    save_plot: bool = False,
    plot_output_dir: Path = Path("Plots"),
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}

    for scenario in scenarios:
        scenario_dir = recorder_data_dir / scenario
        scenario_rows: list[dict[str, float]] = []

        for folder in sorted(scenario_dir.glob("scour_*")):
            try:
                row = process_scour_folder(
                    folder,
                    scenario=scenario,
                    element_ids=element_ids,
                    files=files,
                    num_grid=num_grid,
                    save_plot=save_plot,
                    plot_output_dir=plot_output_dir,
                )
                scenario_rows.append(row)
            except Exception as exc:
                if verbose:
                    print(f"⚠️ Skipped {scenario}/{folder.name}: {exc}")

        if scenario_rows:
            results[scenario] = pd.DataFrame(scenario_rows).sort_values(
                "Scour_Depth_mm"
            )

    return results


def save_results_by_scenario(
    results_by_scenario: dict[str, pd.DataFrame],
    *,
    output_excel_path: Path,
) -> Path:
    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_excel_path) as writer:
        for scenario, df in results_by_scenario.items():
            df.to_excel(writer, sheet_name=scenario, index=False)
    return output_excel_path


def run_batch_workflow(
    *,
    recorder_data_dir: Path = Path("RecorderData"),
    output_excel_path: Path = Path("RecorderData") / "Yield_Results_by_Scenario.xlsx",
    scenarios: Iterable[str] = DEFAULT_SCENARIOS,
    element_ids: Iterable[int] = DEFAULT_ELEMENT_IDS,
    files: RecorderFilesConfig = RecorderFilesConfig(),
    num_grid: int = 200,
    save_plot: bool = False,
    plot_output_dir: Path = Path("Plots"),
    verbose: bool = True,
) -> Path | None:
    results = batch_process_scenarios(
        recorder_data_dir,
        scenarios=scenarios,
        element_ids=element_ids,
        files=files,
        num_grid=num_grid,
        save_plot=save_plot,
        plot_output_dir=plot_output_dir,
        verbose=verbose,
    )

    if not results:
        if verbose:
            print("⚠️ No results to save.")
        return None

    output_path = save_results_by_scenario(results, output_excel_path=output_excel_path)
    if verbose:
        print(f"✅ Saved results to {output_path}")
    return output_path


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract bilinear yield metrics from RecorderData/<Scenario>/scour_* "
            "folders and write RecorderData/Yield_Results_by_Scenario.xlsx"
        )
    )
    parser.add_argument(
        "--recorder-data-dir",
        type=Path,
        default=Path("RecorderData"),
        help="Root directory containing scenario folders (default: RecorderData).",
    )
    parser.add_argument(
        "--output-excel",
        type=Path,
        default=Path("RecorderData") / "Yield_Results_by_Scenario.xlsx",
        help="Output workbook path.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        help="Scenario folder names to process.",
    )
    parser.add_argument(
        "--element-ids",
        nargs="+",
        type=int,
        default=list(DEFAULT_ELEMENT_IDS),
        help="Element IDs used to sum base shear V2_j.",
    )
    parser.add_argument(
        "--displacement-file",
        default="Displacement.5201.out",
        help="Displacement recorder filename within each scour folder.",
    )
    parser.add_argument(
        "--column-displacement-file",
        default="ColDisplacement.3201.out",
        help="Column displacement recorder filename within each scour folder.",
    )
    parser.add_argument(
        "--force-file-template",
        default="ColLocForce.{element_id}.out",
        help="Template for force recorder filenames.",
    )
    parser.add_argument(
        "--moment-element-id",
        type=int,
        default=3201,
        help="Element ID used for moment extraction (M3_j).",
    )
    parser.add_argument(
        "--num-grid",
        type=int,
        default=200,
        help="Number of candidate dy points for bilinear fitting.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save bilinear fit plots to --plot-dir/<Scenario>/.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("Plots"),
        help="Output directory for optional plots.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error console output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    files = RecorderFilesConfig(
        displacement_file=args.displacement_file,
        column_displacement_file=args.column_displacement_file,
        force_file_template=args.force_file_template,
        moment_element_id=args.moment_element_id,
    )

    run_batch_workflow(
        recorder_data_dir=args.recorder_data_dir,
        output_excel_path=args.output_excel,
        scenarios=args.scenarios,
        element_ids=args.element_ids,
        files=files,
        num_grid=args.num_grid,
        save_plot=args.plot,
        plot_output_dir=args.plot_dir,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
