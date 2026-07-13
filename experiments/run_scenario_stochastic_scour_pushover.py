#!/usr/bin/env python
"""Scenario-based stochastic scour/material pushover experiment.

This driver first generates Latin-hypercube samples of scenario scour depth and
material properties. It can stop after saving the stochastic input data and
histogram/CDF plots, or continue through nonlinear OpenSees pushover analyses
and capacity tuple extraction for each sampled row.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.parameters import ANALYSIS, GEOMETRY, MATERIALS, PATHS, SCOUR
from src.postprocessing.bilinear_fit import fit_bilinear_profile
from src.scour import LHS_scour_hazard


DEFAULT_OUTPUT_DIR = Path("experiments") / "output" / "scenario_stochastic_scour_pushover"
DEFAULT_SCENARIOS = list(SCOUR["scenarios"].keys())
NATURE_SCENARIO_COLORS = ("#3B6EA8", "#C47A2C", "#4C8C6A", "#7E5AA7", "#A64B4B")


@dataclass(frozen=True)
class ExperimentConfig:
    scenarios: list[str]
    samples_per_scenario: int
    seed: int | None
    pushlimit_percent: float
    initial_incr_mm: float | None
    min_incr_mm: float | None
    max_incr_mm: float | None
    displacement_control_num_iter: int | None
    output_dir: Path
    input_only: bool
    dry_run: bool
    skip_existing: bool
    no_plots: bool


@dataclass(frozen=True)
class PushoverCurve:
    global_sample_index: int
    scenario_sample_index: int
    scenario_id: int
    scenario: str
    s_z_continuous_m: float
    s_z_model_m: float
    fc_mpa: float
    fy_mpa: float
    capacity_point: tuple[float, float, float, float]
    displacement_mm: np.ndarray
    base_shear_kN: np.ndarray
    rotation_rad: np.ndarray
    base_moment_kNm: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate scenario-based LHS scour/material samples, optionally "
            "plot input histograms only, or run nonlinear pushover for every "
            "sample and save capacity tuples."
        )
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCOUR["scenarios"].keys()),
        default=DEFAULT_SCENARIOS,
        help="Scenario names to simulate.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=int(SCOUR["num_lhs_samples_per_scenario"]),
        help="Latin-hypercube samples per scenario.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--input-only",
        action="store_true",
        help="Only generate stochastic input data and histogram/CDF plots.",
    )
    parser.add_argument(
        "--pushlimit",
        type=float,
        default=ANALYSIS["pushover"]["max_drift_ratio"] * 100.0,
        help="Pushover drift limit as percent of effective bridge height.",
    )
    parser.add_argument(
        "--initial-incr-mm",
        type=float,
        default=None,
        help="Initial adaptive DisplacementControl increment in mm.",
    )
    parser.add_argument(
        "--min-incr-mm",
        type=float,
        default=None,
        help="Minimum adaptive DisplacementControl increment in mm.",
    )
    parser.add_argument(
        "--max-incr-mm",
        type=float,
        default=None,
        help="Maximum adaptive DisplacementControl increment in mm.",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=None,
        help="Desired iteration count for adaptive DisplacementControl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for samples, plots, raw curves, and tuple CSV.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing raw curves and recompute tuple rows from them.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned run without generating samples or running OpenSees.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    args = build_parser().parse_args(argv)
    if args.samples <= 0:
        raise SystemExit("--samples must be positive.")
    if args.pushlimit <= 0:
        raise SystemExit("--pushlimit must be positive.")
    for name in ("initial_incr_mm", "min_incr_mm", "max_incr_mm"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    if (
        args.min_incr_mm is not None
        and args.max_incr_mm is not None
        and args.min_incr_mm > args.max_incr_mm
    ):
        raise SystemExit("--min-incr-mm must be less than or equal to --max-incr-mm.")
    if args.num_iter is not None and args.num_iter <= 0:
        raise SystemExit("--num-iter must be positive.")

    return ExperimentConfig(
        scenarios=list(args.scenarios),
        samples_per_scenario=int(args.samples),
        seed=args.seed,
        pushlimit_percent=float(args.pushlimit),
        initial_incr_mm=None if args.initial_incr_mm is None else float(args.initial_incr_mm),
        min_incr_mm=None if args.min_incr_mm is None else float(args.min_incr_mm),
        max_incr_mm=None if args.max_incr_mm is None else float(args.max_incr_mm),
        displacement_control_num_iter=args.num_iter,
        output_dir=args.output_dir,
        input_only=bool(args.input_only),
        dry_run=bool(args.dry_run),
        skip_existing=bool(args.skip_existing),
        no_plots=bool(args.no_plots),
    )


def output_paths(config: ExperimentConfig) -> dict[str, Path]:
    root = config.output_dir
    return {
        "root": root,
        "samples": root / "samples",
        "plots": root / "plots",
        "material_plots": root / "plots" / "materials",
        "histogram_data": root / "histogram_data",
        "raw_curves": root / "raw_curves",
        "recorders": root / "recorders",
        "capacity_tuples": root / "capacity_tuples.csv",
        "capacity_tuples_progress": root / "capacity_tuples_progress.csv",
        "manifest": root / "manifest.json",
    }


def ensure_output_dirs(paths: dict[str, Path]) -> None:
    for key in (
        "root",
        "samples",
        "plots",
        "material_plots",
        "histogram_data",
        "raw_curves",
        "recorders",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)


def configured_scour_depths() -> np.ndarray:
    return np.asarray(sorted(float(value) for value in SCOUR["spring_removal_depths_m"]))


def discretize_scour_depth_m(value: float) -> float:
    """Map a continuous scour sample to the next configured spring-removal bin."""
    grid = configured_scour_depths()
    if value <= grid[0]:
        return float(grid[0])
    for depth in grid:
        if value <= depth + 1.0e-12:
            return float(depth)
    return float(grid[-1])


def lhs_standard_normal(samples: int, rng: np.random.Generator) -> np.ndarray:
    probabilities = (np.arange(samples, dtype=float) + rng.random(samples)) / samples
    return rng.permutation(norm.ppf(probabilities))


def lhs_material_samples(samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fc_z = lhs_standard_normal(samples, rng)
    fy_z = lhs_standard_normal(samples, rng)
    fc = (
        float(MATERIALS["concrete"]["mean_MPa"])
        + float(MATERIALS["concrete"]["std_MPa"]) * fc_z
    )
    fy_log_mean = math.log(float(MATERIALS["steel"]["mean_MPa"]))
    fy_log_std = float(MATERIALS["steel"]["std_MPa"]) / float(
        MATERIALS["steel"]["mean_MPa"]
    )
    fy = np.exp(fy_log_mean + fy_log_std * fy_z)
    return fc, fy


def simulate_scenario_samples(config: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, dict]]:
    rng = np.random.default_rng(config.seed)
    tables: list[pd.DataFrame] = []
    hazard_stats: dict[str, dict] = {}
    global_index = 0
    scenario_ids = {scenario: index for index, scenario in enumerate(config.scenarios)}

    for scenario in config.scenarios:
        scenario_params = SCOUR["scenarios"][scenario]
        hazard = LHS_scour_hazard(
            lhsN=config.samples_per_scenario,
            vel=scenario_params["velocity_m_s"],
            dPier=GEOMETRY["pier_diameter_m"],
            gama=SCOUR["kinematic_viscosity_m2_s"],
            zDot=scenario_params["erosion_rate_mm_hr"],
            rng=rng,
        )
        fc, fy = lhs_material_samples(config.samples_per_scenario, rng)
        scenario_indices = np.arange(config.samples_per_scenario, dtype=int)
        table = pd.DataFrame(
            {
                "global_sample_index": np.arange(
                    global_index,
                    global_index + config.samples_per_scenario,
                    dtype=int,
                ),
                "scenario_sample_index": scenario_indices,
                "scenario_id": scenario_ids[scenario],
                "scenario": scenario,
                "scenario_description": scenario_params["description"],
                "s_z_continuous_m": np.asarray(hazard["z50Final"], dtype=float),
                "fc_MPa": fc,
                "fy_MPa": fy,
            }
        )
        table["s_z_model_m"] = table["s_z_continuous_m"].apply(discretize_scour_depth_m)
        tables.append(table)
        global_index += config.samples_per_scenario

        hazard_stats[scenario] = {
            "velocity_m_s": float(scenario_params["velocity_m_s"]),
            "erosion_rate_mm_hr": float(scenario_params["erosion_rate_mm_hr"]),
            "z50Mean_m": float(hazard["z50Mean"]),
            "z50std_m": float(hazard["z50std"]),
            "z50LogMean": float(hazard["z50LogMean"]),
            "z50LogStd": float(hazard["z50LogStd"]),
        }

    return pd.concat(tables, ignore_index=True), hazard_stats


def save_sample_tables(samples_df: pd.DataFrame, paths: dict[str, Path]) -> list[Path]:
    output_files: list[Path] = []
    all_samples_path = paths["samples"] / "all_scenario_samples.csv"
    samples_df.to_csv(all_samples_path, index=False)
    output_files.append(all_samples_path)

    for scenario, scenario_df in samples_df.groupby("scenario", sort=True):
        scenario_path = paths["samples"] / f"scenario_{scenario}_samples.csv"
        scenario_df.to_csv(scenario_path, index=False)
        output_files.append(scenario_path)

    return output_files


def scour_counts_for_grid(values: pd.Series) -> pd.Series:
    grid = configured_scour_depths()
    counts = values.value_counts().reindex(grid, fill_value=0)
    counts.index = counts.index.astype(float)
    return counts


def scour_bin_edges() -> np.ndarray:
    """Shared histogram bin edges centered on configured feasible scour depths."""
    grid = configured_scour_depths()
    if len(grid) < 2:
        return np.asarray([grid[0] - 0.25, grid[0] + 0.25], dtype=float)
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    first_edge = grid[0] - 0.5 * (grid[1] - grid[0])
    last_edge = grid[-1] + 0.5 * (grid[-1] - grid[-2])
    return np.concatenate(([first_edge], midpoints, [last_edge])).astype(float)


def apply_nature_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
        }
    )


def scenario_color_map(scenarios: Iterable[str]) -> dict[str, str]:
    return {
        scenario: NATURE_SCENARIO_COLORS[index % len(NATURE_SCENARIO_COLORS)]
        for index, scenario in enumerate(scenarios)
    }


def plot_scour_histograms(samples_df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_nature_plot_style()
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    grouped = list(samples_df.groupby("scenario", sort=False))
    colors = scenario_color_map([str(scenario) for scenario, _ in grouped])

    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    bins = scour_bin_edges()
    for scenario, scenario_df in grouped:
        values = scenario_df["s_z_model_m"].to_numpy(dtype=float)
        ax.hist(
            values,
            bins=bins,
            color=colors[str(scenario)],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.48,
            label=str(scenario),
        )
    ax.set_xlabel("Discrete scour depth, s_z (m)")
    ax.set_ylabel("Frequency")
    ax.set_title("Scour-depth histogram")
    ax.grid(True, axis="y", color="0.9", linewidth=0.5)
    ax.legend(title="Scenario", title_fontsize=6.5)
    fig.tight_layout()
    output_path = plots_dir / "scour_histograms_by_scenario.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    output_files.append(output_path)

    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    for scenario, scenario_df in grouped:
        counts = scour_counts_for_grid(scenario_df["s_z_model_m"])
        depths = counts.index.to_numpy(dtype=float)
        frequencies = counts.to_numpy(dtype=int)
        cumulative = np.cumsum(frequencies) / max(1, frequencies.sum())
        ax.step(
            depths,
            cumulative,
            where="post",
            color=colors[str(scenario)],
            label=str(scenario),
        )
    ax.set_xlabel("Discrete scour depth, s_z (m)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Cumulative scour-depth histogram")
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.legend(title="Scenario", title_fontsize=6.5)
    fig.tight_layout()
    output_path = plots_dir / "scour_cumulative_histograms_by_scenario.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    output_files.append(output_path)

    return output_files


def plot_material_histograms(samples_df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_nature_plot_style()
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    grouped = list(samples_df.groupby("scenario", sort=False))
    colors = scenario_color_map([str(scenario) for scenario, _ in grouped])
    specs = [
        ("fc_MPa", "Concrete compressive strength, fc (MPa)", "fc"),
        ("fy_MPa", "Steel yield strength, fy (MPa)", "fy"),
    ]

    for column, label, stem in specs:
        fig, ax = plt.subplots(figsize=(3.5, 2.55))
        all_values = samples_df[column].to_numpy(dtype=float)
        bins = np.linspace(float(all_values.min()), float(all_values.max()), 31)
        for scenario, scenario_df in grouped:
            values = scenario_df[column].to_numpy(dtype=float)
            ax.hist(
                values,
                bins=bins,
                color=colors[str(scenario)],
                edgecolor="white",
                linewidth=0.25,
                alpha=0.48,
                label=str(scenario),
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{label} histogram")
        ax.grid(True, axis="y", color="0.9", linewidth=0.5)
        ax.legend(title="Scenario", title_fontsize=6.5)
        fig.tight_layout()
        output_path = plots_dir / f"{stem}_histograms_by_scenario.pdf"
        fig.savefig(output_path)
        plt.close(fig)
        output_files.append(output_path)

        fig, ax = plt.subplots(figsize=(3.5, 2.55))
        for scenario, scenario_df in grouped:
            sorted_values = np.sort(scenario_df[column].to_numpy(dtype=float))
            cumulative = (
                np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
            )
            ax.step(
                sorted_values,
                cumulative,
                where="post",
                color=colors[str(scenario)],
                label=str(scenario),
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Cumulative probability")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"{label} cumulative histogram")
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.legend(title="Scenario", title_fontsize=6.5)
        fig.tight_layout()
        output_path = plots_dir / f"{stem}_cumulative_histograms_by_scenario.pdf"
        fig.savefig(output_path)
        plt.close(fig)
        output_files.append(output_path)

    return output_files


def save_histogram_data(samples_df: pd.DataFrame, histogram_dir: Path) -> list[Path]:
    histogram_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []

    for scenario, scenario_df in samples_df.groupby("scenario", sort=True):
        counts = scour_counts_for_grid(scenario_df["s_z_model_m"])
        frequencies = counts.to_numpy(dtype=int)
        cumulative = np.cumsum(frequencies) / max(1, frequencies.sum())
        scour_df = pd.DataFrame(
            {
                "scenario": scenario,
                "s_z_model_m": counts.index.to_numpy(dtype=float),
                "frequency": frequencies,
                "cumulative_probability": cumulative,
            }
        )
        output_path = histogram_dir / f"scenario_{scenario}_scour_histogram_data.csv"
        scour_df.to_csv(output_path, index=False)
        output_files.append(output_path)

    for column in ("fc_MPa", "fy_MPa"):
        sorted_values = np.sort(samples_df[column].to_numpy(dtype=float))
        cumulative = np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
        material_df = pd.DataFrame(
            {
                column: sorted_values,
                "cumulative_probability": cumulative,
            }
        )
        output_path = histogram_dir / f"{column}_empirical_cdf_data.csv"
        material_df.to_csv(output_path, index=False)
        output_files.append(output_path)

    return output_files


def apply_runtime_overrides(config: ExperimentConfig, recorders_dir: Path) -> None:
    PATHS["recorder_data"] = recorders_dir
    pushover = ANALYSIS["pushover"]
    pushover["max_drift_ratio"] = config.pushlimit_percent / 100.0
    if config.initial_incr_mm is not None:
        pushover["displacement_increment_mm"] = config.initial_incr_mm
    if config.min_incr_mm is not None:
        pushover["displacement_increment_min_mm"] = config.min_incr_mm
    if config.max_incr_mm is not None:
        pushover["displacement_increment_max_mm"] = config.max_incr_mm
    if config.displacement_control_num_iter is not None:
        pushover["displacement_control_num_iter"] = (
            config.displacement_control_num_iter
        )


def pushover_step_estimate(config: ExperimentConfig) -> tuple[float, float, int]:
    pushover = ANALYSIS["pushover"]
    effective_height_m = float(pushover.get("effective_bridge_height_m", 13.05))
    initial_incr_mm = float(
        config.initial_incr_mm
        if config.initial_incr_mm is not None
        else pushover.get("displacement_increment_mm", 1.0)
    )
    max_incr_mm = float(
        config.max_incr_mm
        if config.max_incr_mm is not None
        else pushover.get("displacement_increment_max_mm", initial_incr_mm)
    )
    initial_incr_mm = min(initial_incr_mm, max_incr_mm)
    target_mm = config.pushlimit_percent / 100.0 * effective_height_m * 1000.0
    estimated_min_steps = int(math.ceil(target_mm / max_incr_mm - 1.0e-9))
    return target_mm, initial_incr_mm, estimated_min_steps


def raw_curve_path(raw_dir: Path, row: pd.Series) -> Path:
    scenario = str(row["scenario"])
    global_index = int(row["global_sample_index"])
    scenario_index = int(row["scenario_sample_index"])
    scour_depth_m = float(row["s_z_model_m"])
    return (
        raw_dir
        / scenario
        / f"sample_{global_index:06d}_scenario_{scenario_index:05d}_scour_{scour_depth_m:05.2f}m.csv"
    )


def is_terminal_pile_tip_case(scour_depth_m: float) -> bool:
    return math.isclose(scour_depth_m, 20.0, abs_tol=1.0e-9)


def save_terminal_zero_curve(csv_path: Path, row: pd.Series) -> None:
    df = pd.DataFrame(
        {
            "global_sample_index": [int(row["global_sample_index"])],
            "scenario_sample_index": [int(row["scenario_sample_index"])],
            "scenario_id": [int(row["scenario_id"])],
            "scenario": [str(row["scenario"])],
            "s_z_continuous_m": [float(row["s_z_continuous_m"])],
            "s_z_model_m": [float(row["s_z_model_m"])],
            "fc_MPa": [float(row["fc_MPa"])],
            "fy_MPa": [float(row["fy_MPa"])],
            "Delta_mm": [0.0],
            "V_kN": [0.0],
            "theta_x_rad": [0.0],
            "M_x_kNm": [0.0],
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def save_raw_curve(curve: PushoverCurve, csv_path: Path) -> None:
    min_len = min(
        len(curve.displacement_mm),
        len(curve.base_shear_kN),
        len(curve.rotation_rad),
        len(curve.base_moment_kNm),
    )
    df = pd.DataFrame(
        {
            "global_sample_index": [curve.global_sample_index] * min_len,
            "scenario_sample_index": [curve.scenario_sample_index] * min_len,
            "scenario_id": [curve.scenario_id] * min_len,
            "scenario": [curve.scenario] * min_len,
            "s_z_continuous_m": [curve.s_z_continuous_m] * min_len,
            "s_z_model_m": [curve.s_z_model_m] * min_len,
            "fc_MPa": [curve.fc_mpa] * min_len,
            "fy_MPa": [curve.fy_mpa] * min_len,
            "Delta_mm": curve.displacement_mm[:min_len],
            "V_kN": curve.base_shear_kN[:min_len],
            "theta_x_rad": curve.rotation_rad[:min_len],
            "M_x_kNm": curve.base_moment_kNm[:min_len],
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def load_raw_curve(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def capacity_from_raw_curve(df: pd.DataFrame) -> tuple[float, float, float, float] | None:
    if len(df) < 10:
        return None
    required_columns = {"Delta_mm", "V_kN", "M_x_kNm", "theta_x_rad"}
    if not required_columns.issubset(df.columns):
        return None
    displacement_mm = df["Delta_mm"].to_numpy(dtype=float)
    base_shear_kN = df["V_kN"].to_numpy(dtype=float)
    k1, k2, dy_mm, vy_kN = fit_bilinear_profile(displacement_mm, base_shear_kN)
    if k1 is None or k2 is None or dy_mm is None or vy_kN is None:
        return None
    i_yield = int(np.argmin(np.abs(displacement_mm - dy_mm)))
    return (
        float(vy_kN),
        float(dy_mm),
        float(df["M_x_kNm"].iloc[i_yield]),
        float(df["theta_x_rad"].iloc[i_yield]),
    )


def capacity_row(
    row: pd.Series,
    *,
    status: str,
    n_points: int = 0,
    capacity_point: tuple[float, float, float, float] | None = None,
    target_displacement_mm: float | None = None,
    max_displacement_mm: float | None = None,
    target_reached: bool | None = None,
    raw_curve_file: Path | None = None,
) -> dict[str, float | int | str | bool | None]:
    if capacity_point is None:
        vy, dy, my, theta = (None, None, None, None)
    else:
        vy, dy, my, theta = capacity_point
    return {
        "global_sample_index": int(row["global_sample_index"]),
        "scenario_sample_index": int(row["scenario_sample_index"]),
        "scenario_id": int(row["scenario_id"]),
        "scenario": str(row["scenario"]),
        "s_z_continuous_m": float(row["s_z_continuous_m"]),
        "s_z_model_m": float(row["s_z_model_m"]),
        "fc_MPa": float(row["fc_MPa"]),
        "fy_MPa": float(row["fy_MPa"]),
        "status": status,
        "n_points": n_points,
        "target_displacement_mm": target_displacement_mm,
        "max_displacement_mm": max_displacement_mm,
        "target_reached": target_reached,
        "V_y_kN": vy,
        "Delta_y_mm": dy,
        "M_x_kNm": my,
        "theta_x_rad": theta,
        "raw_curve_file": "" if raw_curve_file is None else str(raw_curve_file),
    }


def run_one_sample(
    config: ExperimentConfig,
    row: pd.Series,
    recorders_dir: Path,
) -> PushoverCurve | None:
    from BridgeModeling.Pushover import run_single_pushover_simulation

    apply_runtime_overrides(config, recorders_dir)
    result = run_single_pushover_simulation(
        scenario=str(row["scenario"]),
        random_seed=None,
        scour_depth_m=float(row["s_z_model_m"]),
        fc_MPa=float(row["fc_MPa"]),
        fy_MPa=float(row["fy_MPa"]),
    )
    if result is None:
        return None

    capacity_point, displacement_mm, base_shear_kN, rotation_rad, base_moment_kNm = result
    return PushoverCurve(
        global_sample_index=int(row["global_sample_index"]),
        scenario_sample_index=int(row["scenario_sample_index"]),
        scenario_id=int(row["scenario_id"]),
        scenario=str(row["scenario"]),
        s_z_continuous_m=float(row["s_z_continuous_m"]),
        s_z_model_m=float(row["s_z_model_m"]),
        fc_mpa=float(row["fc_MPa"]),
        fy_mpa=float(row["fy_MPa"]),
        capacity_point=tuple(float(value) for value in capacity_point),
        displacement_mm=np.asarray(displacement_mm, dtype=float),
        base_shear_kN=np.asarray(base_shear_kN, dtype=float),
        rotation_rad=np.asarray(rotation_rad, dtype=float),
        base_moment_kNm=np.asarray(base_moment_kNm, dtype=float),
    )


def save_manifest(
    config: ExperimentConfig,
    paths: dict[str, Path],
    hazard_stats: dict[str, dict] | None = None,
) -> None:
    target_mm, initial_incr_mm, estimated_steps = pushover_step_estimate(config)
    expected_total_rows = len(config.scenarios) * config.samples_per_scenario
    manifest = {
        "purpose": "scenario-based stochastic scour/material pushover experiment",
        "scenarios": config.scenarios,
        "samples_per_scenario": config.samples_per_scenario,
        "expected_total_rows": expected_total_rows,
        "seed": config.seed,
        "sampling": {
            "scour": "LHS_scour_hazard by flooding scenario",
            "materials": "Latin-hypercube normal-space samples",
            "scour_discretization": (
                "ceil continuous s_z to the next configured spring-removal depth"
            ),
            "spring_removal_depths_m": configured_scour_depths().tolist(),
        },
        "spring_removal_scope": "all pile springs across all three pile groups/bents",
        "terminal_note": (
            "20 m represents physical scour to the pile tip and is recorded as "
            "a zero-capacity terminal state without launching OpenSees."
        ),
        "pushlimit_percent": config.pushlimit_percent,
        "target_displacement_mm": target_mm,
        "initial_increment_mm": initial_incr_mm,
        "min_increment_mm": config.min_incr_mm,
        "max_increment_mm": config.max_incr_mm,
        "displacement_control_num_iter": config.displacement_control_num_iter,
        "estimated_minimum_pushover_steps": estimated_steps,
        "input_only": config.input_only,
        "outputs": {key: str(value) for key, value in paths.items()},
        "hazard_stats": hazard_stats or {},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def print_plan(config: ExperimentConfig, paths: dict[str, Path]) -> None:
    target_mm, initial_incr_mm, estimated_steps = pushover_step_estimate(config)
    total_samples = len(config.scenarios) * config.samples_per_scenario
    print("Scenario stochastic scour/material pushover experiment")
    print(f"  scenarios: {', '.join(config.scenarios)}")
    print(f"  LHS samples per scenario: {config.samples_per_scenario}")
    print(f"  total stochastic rows: {total_samples}")
    print(
        "  scour bins s_z (m): "
        f"{', '.join(f'{value:g}' for value in configured_scour_depths())}"
    )
    print(f"  input-only: {config.input_only}")
    print(f"  pushlimit: {config.pushlimit_percent:g}% of effective bridge height")
    print(
        "  pushover target: "
        f"{target_mm:g} mm in at least ~{estimated_steps} steps "
        f"(initial {initial_incr_mm:g} mm/step)"
    )
    print(f"  output root: {paths['root']}")
    print(f"  capacity tuples: {paths['capacity_tuples']}")
    print(f"  progress tuples: {paths['capacity_tuples_progress']}")


def write_capacity_progress(
    rows: list[dict[str, float | int | str | bool | None]],
    paths: dict[str, Path],
) -> None:
    pd.DataFrame(rows).to_csv(paths["capacity_tuples_progress"], index=False)


def run_pushover_sweep(
    config: ExperimentConfig,
    paths: dict[str, Path],
    samples_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool | None]] = []
    target_mm, _, _ = pushover_step_estimate(config)

    for _, row in samples_df.iterrows():
        csv_path = raw_curve_path(paths["raw_curves"], row)
        scour_depth_m = float(row["s_z_model_m"])
        label = (
            f"{row['scenario']} sample {int(row['scenario_sample_index'])} "
            f"(global {int(row['global_sample_index'])}, s_z={scour_depth_m:g} m)"
        )

        if is_terminal_pile_tip_case(scour_depth_m):
            save_terminal_zero_curve(csv_path, row)
            rows.append(
                capacity_row(
                    row,
                    status="terminal_pile_tip_zero_capacity",
                    n_points=1,
                    capacity_point=(0.0, 0.0, 0.0, 0.0),
                    target_displacement_mm=target_mm,
                    max_displacement_mm=0.0,
                    target_reached=False,
                    raw_curve_file=csv_path,
                )
            )
            print(f"Recorded terminal pile-tip case: {label}")
            write_capacity_progress(rows, paths)
            continue

        if config.skip_existing and csv_path.exists():
            df_existing = load_raw_curve(csv_path)
            max_disp = (
                float(df_existing["Delta_mm"].max()) if "Delta_mm" in df_existing else None
            )
            target_reached = None if max_disp is None else max_disp >= target_mm - 1.0e-6
            status = (
                "skipped_existing_ok"
                if target_reached
                else "skipped_existing_partial"
                if target_reached is False
                else "skipped_existing"
            )
            rows.append(
                capacity_row(
                    row,
                    status=status,
                    n_points=len(df_existing),
                    capacity_point=capacity_from_raw_curve(df_existing),
                    target_displacement_mm=target_mm,
                    max_displacement_mm=max_disp,
                    target_reached=target_reached,
                    raw_curve_file=csv_path,
                )
            )
            print(f"Skipping existing {label}: {csv_path}")
            write_capacity_progress(rows, paths)
            continue

        print(f"\nRunning nonlinear pushover for {label}...")
        try:
            sample_recorders_dir = paths["recorders"] / f"sample_{int(row['global_sample_index']):06d}"
            curve = run_one_sample(config, row, sample_recorders_dir)
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"Required runtime dependency is missing: {exc}")
            raise

        if curve is None:
            rows.append(
                capacity_row(
                    row,
                    status="failed",
                    target_displacement_mm=target_mm,
                    target_reached=False,
                    raw_curve_file=csv_path,
                )
            )
            print(f"Simulation failed for {label}.")
            write_capacity_progress(rows, paths)
            continue

        save_raw_curve(curve, csv_path)
        max_disp = float(np.max(curve.displacement_mm))
        target_reached = max_disp >= target_mm - 1.0e-6
        rows.append(
            capacity_row(
                row,
                status="ok" if target_reached else "partial",
                n_points=len(curve.displacement_mm),
                capacity_point=curve.capacity_point,
                target_displacement_mm=target_mm,
                max_displacement_mm=max_disp,
                target_reached=target_reached,
                raw_curve_file=csv_path,
            )
        )
        print(f"Saved {'raw' if target_reached else 'partial raw'} curve: {csv_path}")
        write_capacity_progress(rows, paths)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    paths = output_paths(config)
    print_plan(config, paths)

    if config.dry_run:
        print("Dry run only; no samples were generated and OpenSeesPy was not imported.")
        return 0

    ensure_output_dirs(paths)
    samples_df, hazard_stats = simulate_scenario_samples(config)
    sample_files = save_sample_tables(samples_df, paths)
    histogram_data_files = save_histogram_data(samples_df, paths["histogram_data"])
    save_manifest(config, paths, hazard_stats)
    for sample_file in sample_files:
        print(f"Saved sample data: {sample_file}")
    for histogram_data_file in histogram_data_files:
        print(f"Saved histogram data: {histogram_data_file}")

    if not config.no_plots:
        plot_files = []
        plot_files.extend(plot_scour_histograms(samples_df, paths["plots"]))
        plot_files.extend(plot_material_histograms(samples_df, paths["material_plots"]))
        for plot_file in plot_files:
            print(f"Saved plot: {plot_file}")

    if config.input_only:
        print("Input-only run complete; nonlinear pushover was not executed.")
        return 0

    try:
        capacity_df = run_pushover_sweep(config, paths, samples_df)
    except (ImportError, ModuleNotFoundError):
        return 2

    capacity_df.to_csv(paths["capacity_tuples"], index=False)
    print(f"\nSaved capacity tuples: {paths['capacity_tuples']}")
    expected_total_rows = len(config.scenarios) * config.samples_per_scenario
    if len(capacity_df) != expected_total_rows:
        print(
            "Capacity tuple row-count mismatch: "
            f"expected {expected_total_rows}, got {len(capacity_df)}."
        )
        return 1
    if capacity_df.empty or (capacity_df["status"] == "failed").all():
        print("No successful, partial, or terminal simulations completed.")
        return 1
    print(f"Final capacity tuple rows: {len(capacity_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
