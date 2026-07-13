#!/usr/bin/env python
"""Run deterministic scour-depth pushover sweeps.

This is a manual, long-running test/experiment runner. It executes the real
OpenSeesPy bridge model for deterministic realized scour depths s_z and saves
the full nonlinear pushover curves for later bilinear-fit studies.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT_DIR = Path("tests") / "output" / "deterministic_scour_pushover"


@dataclass(frozen=True)
class SweepConfig:
    scenario: str
    min_scour_m: float
    max_scour_m: float
    step_m: float
    pushlimit_percent: float
    initial_incr_mm: float | None
    min_incr_mm: float | None
    max_incr_mm: float | None
    displacement_control_num_iter: int | None
    fc_mpa: float
    fy_mpa: float
    output_dir: Path
    skip_existing: bool
    dry_run: bool


@dataclass(frozen=True)
class PushoverCurve:
    scour_depth_m: float
    capacity_point: tuple[float, float, float, float]
    displacement_mm: np.ndarray
    base_shear_kN: np.ndarray
    rotation_rad: np.ndarray
    base_moment_kNm: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic OpenSeesPy pushover analyses for realized "
            "scour depths s_z and save raw nonlinear curves."
        )
    )
    parser.add_argument(
        "--scenario",
        default="missouri",
        choices=("missouri", "colorado", "extreme"),
        help=(
            "Scenario label used only for existing output folder conventions. "
            "The scour depths are deterministic."
        ),
    )
    parser.add_argument(
        "--min-scour",
        type=float,
        default=0.0,
        help="Minimum realized scour depth s_z in meters.",
    )
    parser.add_argument(
        "--max-scour",
        type=float,
        default=17.0,
        help="Maximum realized scour depth s_z in meters.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="Scour-depth increment in meters.",
    )
    parser.add_argument(
        "--pushlimit",
        type=float,
        default=5.0,
        help=(
            "Pushover drift limit as a percentage of effective bridge height. "
            "For example, --pushlimit 10 runs to 10%%. Default: 5."
        ),
    )
    parser.add_argument(
        "--initial-incr-mm",
        type=float,
        default=None,
        help="Initial DisplacementControl increment at the control node in mm.",
    )
    parser.add_argument(
        "--min-incr-mm",
        type=float,
        default=None,
        help="Minimum adaptive DisplacementControl increment at the control node in mm.",
    )
    parser.add_argument(
        "--max-incr-mm",
        type=float,
        default=None,
        help="Maximum adaptive DisplacementControl increment at the control node in mm.",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=None,
        help="Desired iteration count used by adaptive DisplacementControl.",
    )
    parser.add_argument(
        "--fc-mpa",
        type=float,
        default=27.0,
        help="Deterministic concrete compressive strength in MPa.",
    )
    parser.add_argument(
        "--fy-mpa",
        type=float,
        default=420.0,
        help="Deterministic steel yield strength in MPa.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for recorder files, raw curves, plots, and summary.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a scour depth when its raw-curve CSV already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned sweep without importing or running OpenSeesPy.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> SweepConfig:
    args = build_parser().parse_args(argv)
    if args.step <= 0:
        raise SystemExit("--step must be positive.")
    if args.max_scour < args.min_scour:
        raise SystemExit("--max-scour must be greater than or equal to --min-scour.")
    if args.pushlimit <= 0:
        raise SystemExit("--pushlimit must be positive.")
    for name in ("initial_incr_mm", "min_incr_mm", "max_incr_mm"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    if args.num_iter is not None and args.num_iter <= 0:
        raise SystemExit("--num-iter must be positive.")
    if (
        args.min_incr_mm is not None
        and args.max_incr_mm is not None
        and args.min_incr_mm > args.max_incr_mm
    ):
        raise SystemExit("--min-incr-mm must be less than or equal to --max-incr-mm.")

    return SweepConfig(
        scenario=args.scenario,
        min_scour_m=float(args.min_scour),
        max_scour_m=float(args.max_scour),
        step_m=float(args.step),
        pushlimit_percent=float(args.pushlimit),
        initial_incr_mm=None if args.initial_incr_mm is None else float(args.initial_incr_mm),
        min_incr_mm=None if args.min_incr_mm is None else float(args.min_incr_mm),
        max_incr_mm=None if args.max_incr_mm is None else float(args.max_incr_mm),
        displacement_control_num_iter=args.num_iter,
        fc_mpa=float(args.fc_mpa),
        fy_mpa=float(args.fy_mpa),
        output_dir=args.output_dir,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )


def scour_depths(config: SweepConfig) -> list[float]:
    depths: list[float] = []
    current = config.min_scour_m
    while current <= config.max_scour_m + 1.0e-12:
        depths.append(round(current, 10))
        current += config.step_m
    if not math.isclose(depths[-1], config.max_scour_m):
        depths.append(round(config.max_scour_m, 10))
    return depths


def output_paths(config: SweepConfig) -> dict[str, Path]:
    root = config.output_dir
    return {
        "root": root,
        "recorders": root / "recorders",
        "raw_curves": root / "raw_curves",
        "plots": root / "plots",
        "summary": root / "capacity_summary.csv",
        "manifest": root / "manifest.json",
    }


def raw_curve_path(raw_dir: Path, scour_depth_m: float) -> Path:
    return raw_dir / f"scour_{scour_depth_m:05.2f}m.csv"


def ensure_output_dirs(paths: dict[str, Path]) -> None:
    for key in ("root", "recorders", "raw_curves", "plots"):
        paths[key].mkdir(parents=True, exist_ok=True)


def redirect_recorder_root(recorders_dir: Path) -> None:
    """Point the existing model output helper at this script's output folder."""
    from config.parameters import PATHS

    PATHS["recorder_data"] = recorders_dir


def pushover_step_estimate(config: SweepConfig) -> tuple[float, float, int]:
    from config.parameters import ANALYSIS

    pushover_config = ANALYSIS["pushover"]
    effective_bridge_height_m = pushover_config.get("effective_bridge_height_m", 13.05)
    default_increment_mm = 0.05 * effective_bridge_height_m * 1000.0 / 100.0
    max_increment_mm = pushover_config.get(
        "displacement_increment_max_mm",
        pushover_config.get("displacement_increment_mm", default_increment_mm),
    )
    if config.max_incr_mm is not None:
        max_increment_mm = config.max_incr_mm
    initial_increment_mm = pushover_config.get(
        "displacement_increment_mm", default_increment_mm
    )
    if config.initial_incr_mm is not None:
        initial_increment_mm = config.initial_incr_mm
    target_displacement_mm = (
        config.pushlimit_percent / 100.0 * effective_bridge_height_m * 1000.0
    )
    estimated_steps = int(
        math.ceil(target_displacement_mm / max_increment_mm - 1.0e-9)
    )
    return target_displacement_mm, initial_increment_mm, estimated_steps


def apply_pushover_overrides(config: SweepConfig) -> None:
    """Override the pushover drift limit for this deterministic test run."""
    from config.parameters import ANALYSIS

    pushover_config = ANALYSIS["pushover"]
    pushover_config["max_drift_ratio"] = config.pushlimit_percent / 100.0
    if config.initial_incr_mm is not None:
        pushover_config["displacement_increment_mm"] = config.initial_incr_mm
    if config.min_incr_mm is not None:
        pushover_config["displacement_increment_min_mm"] = config.min_incr_mm
    if config.max_incr_mm is not None:
        pushover_config["displacement_increment_max_mm"] = config.max_incr_mm
    if config.displacement_control_num_iter is not None:
        pushover_config["displacement_control_num_iter"] = (
            config.displacement_control_num_iter
        )


def run_one_depth(config: SweepConfig, scour_depth_m: float) -> PushoverCurve | None:
    import numpy as np

    redirect_recorder_root(output_paths(config)["recorders"])
    apply_pushover_overrides(config)

    from BridgeModeling.Pushover import run_single_pushover_simulation

    result = run_single_pushover_simulation(
        scenario=config.scenario,
        random_seed=None,
        scour_depth_m=scour_depth_m,
        fc_MPa=config.fc_mpa,
        fy_MPa=config.fy_mpa,
    )
    if result is None:
        return None

    capacity_point, displacement_mm, base_shear_kN, rotation_rad, base_moment_kNm = result
    return PushoverCurve(
        scour_depth_m=scour_depth_m,
        capacity_point=tuple(float(v) for v in capacity_point),
        displacement_mm=np.asarray(displacement_mm, dtype=float),
        base_shear_kN=np.asarray(base_shear_kN, dtype=float),
        rotation_rad=np.asarray(rotation_rad, dtype=float),
        base_moment_kNm=np.asarray(base_moment_kNm, dtype=float),
    )


def save_raw_curve(curve: PushoverCurve, csv_path: Path, pushlimit_percent: float) -> None:
    import pandas as pd

    min_len = min(
        len(curve.displacement_mm),
        len(curve.base_shear_kN),
        len(curve.rotation_rad),
        len(curve.base_moment_kNm),
    )
    df = pd.DataFrame(
        {
            "s_z_m": [curve.scour_depth_m] * min_len,
            "pushlimit_percent": [pushlimit_percent] * min_len,
            "Delta_mm": curve.displacement_mm[:min_len],
            "V_kN": curve.base_shear_kN[:min_len],
            "theta_x_rad": curve.rotation_rad[:min_len],
            "M_x_kNm": curve.base_moment_kNm[:min_len],
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def load_raw_curve(csv_path: Path):
    import pandas as pd

    return pd.read_csv(csv_path)


def capacity_from_raw_curve(df) -> tuple[float, float, float, float] | None:
    if len(df) < 10:
        return None
    required_columns = {"Delta_mm", "V_kN", "M_x_kNm", "theta_x_rad"}
    if not required_columns.issubset(df.columns):
        return None

    import numpy as np
    from src.postprocessing.bilinear_fit import fit_bilinear_profile

    displacement_mm = df["Delta_mm"].to_numpy(dtype=float)
    base_shear_kN = df["V_kN"].to_numpy(dtype=float)
    k1, k2, dy_mm, vy_kN = fit_bilinear_profile(displacement_mm, base_shear_kN)
    if dy_mm is None or vy_kN is None:
        return None

    i_yield = int(np.argmin(np.abs(displacement_mm - dy_mm)))
    my_kNm = float(df["M_x_kNm"].iloc[i_yield])
    thy_rad = float(df["theta_x_rad"].iloc[i_yield])
    return float(vy_kN), float(dy_mm), my_kNm, thy_rad


def curve_summary_row(
    *,
    scour_depth_m: float,
    pushlimit_percent: float,
    status: str,
    n_points: int = 0,
    capacity_point: tuple[float, float, float, float] | None = None,
    target_displacement_mm: float | None = None,
    max_displacement_mm: float | None = None,
    target_reached: bool | None = None,
    raw_curve_file: Path | None = None,
) -> dict[str, float | int | str | None]:
    if capacity_point is None:
        vy, dy, my, thy = (None, None, None, None)
    else:
        vy, dy, my, thy = capacity_point
    return {
        "s_z_m": scour_depth_m,
        "pushlimit_percent": pushlimit_percent,
        "status": status,
        "n_points": n_points,
        "target_displacement_mm": target_displacement_mm,
        "max_displacement_mm": max_displacement_mm,
        "target_reached": target_reached,
        "V_y_kN": vy,
        "Delta_y_mm": dy,
        "M_x_kNm": my,
        "theta_x_rad": thy,
        "raw_curve_file": "" if raw_curve_file is None else str(raw_curve_file),
    }


def save_summary(rows: list[dict[str, float | int | str | None]], summary_path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)


def save_manifest(config: SweepConfig, paths: dict[str, Path], depths: list[float]) -> None:
    target_displacement_mm, initial_increment_mm, estimated_steps = pushover_step_estimate(
        config
    )
    manifest = {
        "purpose": "deterministic scour-depth nonlinear pushover sweep",
        "scenario_label": config.scenario,
        "scour_depths_m": depths,
        "pushlimit_percent": config.pushlimit_percent,
        "pushlimit_ratio": config.pushlimit_percent / 100.0,
        "target_displacement_mm": target_displacement_mm,
        "initial_increment_mm": initial_increment_mm,
        "min_increment_mm": config.min_incr_mm,
        "max_increment_mm": config.max_incr_mm,
        "displacement_control_num_iter": config.displacement_control_num_iter,
        "estimated_minimum_pushover_steps": estimated_steps,
        "fc_mpa": config.fc_mpa,
        "fy_mpa": config.fy_mpa,
        "outputs": {key: str(value) for key, value in paths.items()},
        "notes": [
            "s_z is a realized scour depth measured downward from the original riverbed.",
            "Raw curves are saved before any future bilinear-fit sensitivity study.",
            "Recorder files are redirected under this output directory.",
        ],
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def plot_combined_curves(raw_curve_files: Iterable[Path], plots_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install matplotlib or inspect "
            "the saved raw CSV curves directly."
        ) from exc

    curves: list[pd.DataFrame] = []
    for file_path in raw_curve_files:
        if file_path.exists():
            df = load_raw_curve(file_path)
            if not df.empty:
                curves.append(df)

    if not curves:
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis", len(curves))
    output_files: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, df in enumerate(curves):
        scour_m = float(df["s_z_m"].iloc[0])
        ax.plot(
            df["Delta_mm"],
            df["V_kN"],
            color=cmap(idx),
            linewidth=1.4,
            label=f"{scour_m:g} m",
        )
    ax.set_xlabel("Deck displacement, Delta (mm)")
    ax.set_ylabel("Base shear, V (kN)")
    ax.set_title("Nonlinear pushover curves by deterministic scour depth")
    ax.grid(True, alpha=0.3)
    ax.legend(title="s_z", ncol=2, fontsize=8)
    fig.tight_layout()
    dv_path = plots_dir / "V_vs_Delta_all_scour.png"
    fig.savefig(dv_path, dpi=300)
    plt.close(fig)
    output_files.append(dv_path)

    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, df in enumerate(curves):
        scour_m = float(df["s_z_m"].iloc[0])
        ax.plot(
            df["theta_x_rad"],
            df["M_x_kNm"],
            color=cmap(idx),
            linewidth=1.4,
            label=f"{scour_m:g} m",
        )
    ax.set_xlabel("Column rotation, theta_x (rad)")
    ax.set_ylabel("Base moment, M_x (kNm)")
    ax.set_title("Moment-rotation curves by deterministic scour depth")
    ax.grid(True, alpha=0.3)
    ax.legend(title="s_z", ncol=2, fontsize=8)
    fig.tight_layout()
    mt_path = plots_dir / "Mx_vs_thetax_all_scour.png"
    fig.savefig(mt_path, dpi=300)
    plt.close(fig)
    output_files.append(mt_path)

    return output_files


def print_plan(config: SweepConfig, depths: list[float], paths: dict[str, Path]) -> None:
    target_displacement_mm, initial_increment_mm, estimated_steps = pushover_step_estimate(
        config
    )
    print("Deterministic scour pushover sweep")
    print(f"  scenario label: {config.scenario}")
    print(f"  material values: fc={config.fc_mpa:g} MPa, fy={config.fy_mpa:g} MPa")
    print(f"  pushover pushlimit: {config.pushlimit_percent:g}% of effective bridge height")
    print(
        "  pushover target: "
        f"{target_displacement_mm:g} mm in at least ~{estimated_steps} steps "
        f"(initial {initial_increment_mm:g} mm/step)"
    )
    if any(
        value is not None
        for value in (
            config.min_incr_mm,
            config.max_incr_mm,
            config.displacement_control_num_iter,
        )
    ):
        print(
            "  adaptive override: "
            f"min={config.min_incr_mm}, max={config.max_incr_mm}, "
            f"numIter={config.displacement_control_num_iter}"
        )
    print(f"  scour depths s_z (m): {', '.join(f'{d:g}' for d in depths)}")
    print(f"  output root: {paths['root']}")
    print(f"  raw curves: {paths['raw_curves']}")
    print(f"  plots: {paths['plots']}")
    print(f"  summary: {paths['summary']}")


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    paths = output_paths(config)
    depths = scour_depths(config)
    print_plan(config, depths, paths)

    if config.dry_run:
        print("Dry run only; no OpenSeesPy model was imported or executed.")
        return 0

    ensure_output_dirs(paths)
    save_manifest(config, paths, depths)

    summary_rows: list[dict[str, float | int | str | None]] = []
    raw_files_for_plot: list[Path] = []

    for scour_m in depths:
        csv_path = raw_curve_path(paths["raw_curves"], scour_m)
        raw_files_for_plot.append(csv_path)
        target_displacement_mm, _, _ = pushover_step_estimate(config)

        if config.skip_existing and csv_path.exists():
            df_existing = load_raw_curve(csv_path)
            max_displacement_mm = (
                float(df_existing["Delta_mm"].max()) if "Delta_mm" in df_existing else None
            )
            target_reached = (
                None
                if max_displacement_mm is None
                else max_displacement_mm >= target_displacement_mm - 1.0e-6
            )
            status = (
                "skipped_existing"
                if target_reached is None
                else "skipped_existing_ok"
                if target_reached
                else "skipped_existing_partial"
            )
            capacity_point = capacity_from_raw_curve(df_existing)
            summary_rows.append(
                curve_summary_row(
                    scour_depth_m=scour_m,
                    pushlimit_percent=config.pushlimit_percent,
                    status=status,
                    n_points=len(df_existing),
                    capacity_point=capacity_point,
                    target_displacement_mm=target_displacement_mm,
                    max_displacement_mm=max_displacement_mm,
                    target_reached=target_reached,
                    raw_curve_file=csv_path,
                )
            )
            print(f"Skipping existing s_z={scour_m:g} m: {csv_path}")
            continue

        print(f"\nRunning nonlinear pushover for s_z={scour_m:g} m...")
        try:
            curve = run_one_depth(config, scour_m)
        except (ImportError, ModuleNotFoundError) as exc:
            print(
                "Required runtime dependency is missing for the real OpenSees "
                f"pushover run: {exc}"
            )
            print(
                "Install the project requirements in the active Python environment, "
                "then rerun this script."
            )
            return 2

        if curve is None:
            summary_rows.append(
                curve_summary_row(
                    scour_depth_m=scour_m,
                    pushlimit_percent=config.pushlimit_percent,
                    status="failed",
                    target_displacement_mm=target_displacement_mm,
                    target_reached=False,
                )
            )
            print(f"Simulation failed for s_z={scour_m:g} m.")
            continue

        save_raw_curve(curve, csv_path, config.pushlimit_percent)
        max_displacement_mm = float(max(curve.displacement_mm))
        target_reached = max_displacement_mm >= target_displacement_mm - 1.0e-6
        status = "ok" if target_reached else "partial"
        summary_rows.append(
            curve_summary_row(
                scour_depth_m=scour_m,
                pushlimit_percent=config.pushlimit_percent,
                status=status,
                n_points=len(curve.displacement_mm),
                capacity_point=curve.capacity_point,
                target_displacement_mm=target_displacement_mm,
                max_displacement_mm=max_displacement_mm,
                target_reached=target_reached,
                raw_curve_file=csv_path,
            )
        )
        if target_reached:
            print(f"Saved raw curve: {csv_path}")
        else:
            print(
                f"Saved partial raw curve: {csv_path} "
                f"({max_displacement_mm:.3f}/{target_displacement_mm:.1f} mm)"
            )

    save_summary(summary_rows, paths["summary"])
    print(f"\nSaved summary: {paths['summary']}")

    try:
        plot_files = plot_combined_curves(raw_files_for_plot, paths["plots"])
    except RuntimeError as exc:
        print(f"Plotting skipped: {exc}")
        plot_files = []

    for plot_file in plot_files:
        print(f"Saved plot: {plot_file}")

    successful = sum(1 for row in summary_rows if row["status"] != "failed")
    if successful == 0:
        print("No successful simulations completed.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
