#!/usr/bin/env python
"""Run deterministic pushover over the spring-removal scour grid.

This experiment evaluates the nonlinear OpenSeesPy bridge model at every
configured discrete scour depth whose value changes the modeled pile-soil spring
removal state. Material uncertainty is intentionally disabled; nominal material
values are taken from config.parameters.MATERIALS.
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.parameters import ANALYSIS, MATERIALS, PATHS, SCOUR
from src.postprocessing.bilinear_fit import fit_bilinear_profile


DEFAULT_OUTPUT_DIR = Path("experiments") / "output" / "deterministic_scour_pushover"


@dataclass(frozen=True)
class ExperimentConfig:
    scenario: str
    scour_depths_m: list[float]
    pushlimit_percent: float
    initial_incr_mm: float | None
    min_incr_mm: float | None
    max_incr_mm: float | None
    displacement_control_num_iter: int | None
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


def nominal_materials() -> tuple[float, float]:
    """Return nominal material values from config without stochastic sampling."""
    return (
        float(MATERIALS["concrete"]["mean_MPa"]),
        float(MATERIALS["steel"]["mean_MPa"]),
    )


def configured_scour_depths() -> list[float]:
    return [float(value) for value in SCOUR["spring_removal_depths_m"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic nonlinear pushover analyses at all configured "
            "pile-spring-removal scour depths using nominal materials."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCOUR["scenarios"].keys()),
        default="missouri",
        help="Scenario label used only for recorder/output folder organization.",
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
        help="Root directory for raw curves, recorder files, tuple CSV, and plots.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing raw curves and recompute tuple summaries from them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned experiment without importing or running OpenSeesPy.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    args = build_parser().parse_args(argv)
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
        scenario=args.scenario,
        scour_depths_m=configured_scour_depths(),
        pushlimit_percent=float(args.pushlimit),
        initial_incr_mm=None if args.initial_incr_mm is None else float(args.initial_incr_mm),
        min_incr_mm=None if args.min_incr_mm is None else float(args.min_incr_mm),
        max_incr_mm=None if args.max_incr_mm is None else float(args.max_incr_mm),
        displacement_control_num_iter=args.num_iter,
        output_dir=args.output_dir,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )


def output_paths(config: ExperimentConfig) -> dict[str, Path]:
    root = config.output_dir
    return {
        "root": root,
        "recorders": root / "recorders",
        "raw_curves": root / "raw_curves",
        "plots": root / "plots",
        "capacity_tuples": root / "capacity_tuples.csv",
        "manifest": root / "manifest.json",
    }


def ensure_output_dirs(paths: dict[str, Path]) -> None:
    for key in ("root", "recorders", "raw_curves", "plots"):
        paths[key].mkdir(parents=True, exist_ok=True)


def raw_curve_path(raw_dir: Path, scour_depth_m: float) -> Path:
    return raw_dir / f"scour_{scour_depth_m:05.2f}m.csv"


def is_terminal_pile_tip_case(scour_depth_m: float) -> bool:
    return math.isclose(scour_depth_m, 20.0, abs_tol=1.0e-9)


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


def run_one_depth(config: ExperimentConfig, scour_depth_m: float) -> PushoverCurve | None:
    from BridgeModeling.Pushover import run_single_pushover_simulation

    fc_mpa, fy_mpa = nominal_materials()
    apply_runtime_overrides(config, output_paths(config)["recorders"])
    result = run_single_pushover_simulation(
        scenario=config.scenario,
        random_seed=None,
        scour_depth_m=scour_depth_m,
        fc_MPa=fc_mpa,
        fy_MPa=fy_mpa,
    )
    if result is None:
        return None

    capacity_point, displacement_mm, base_shear_kN, rotation_rad, base_moment_kNm = result
    return PushoverCurve(
        scour_depth_m=scour_depth_m,
        capacity_point=tuple(float(value) for value in capacity_point),
        displacement_mm=np.asarray(displacement_mm, dtype=float),
        base_shear_kN=np.asarray(base_shear_kN, dtype=float),
        rotation_rad=np.asarray(rotation_rad, dtype=float),
        base_moment_kNm=np.asarray(base_moment_kNm, dtype=float),
    )


def save_raw_curve(curve: PushoverCurve, csv_path: Path) -> None:
    fc_mpa, fy_mpa = nominal_materials()
    min_len = min(
        len(curve.displacement_mm),
        len(curve.base_shear_kN),
        len(curve.rotation_rad),
        len(curve.base_moment_kNm),
    )
    df = pd.DataFrame(
        {
            "s_z_m": [curve.scour_depth_m] * min_len,
            "fc_MPa": [fc_mpa] * min_len,
            "fy_MPa": [fy_mpa] * min_len,
            "Delta_mm": curve.displacement_mm[:min_len],
            "V_kN": curve.base_shear_kN[:min_len],
            "theta_x_rad": curve.rotation_rad[:min_len],
            "M_x_kNm": curve.base_moment_kNm[:min_len],
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def save_terminal_zero_curve(csv_path: Path, scour_depth_m: float) -> None:
    fc_mpa, fy_mpa = nominal_materials()
    df = pd.DataFrame(
        {
            "s_z_m": [scour_depth_m],
            "fc_MPa": [fc_mpa],
            "fy_MPa": [fy_mpa],
            "Delta_mm": [0.0],
            "V_kN": [0.0],
            "theta_x_rad": [0.0],
            "M_x_kNm": [0.0],
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
    if dy_mm is None or vy_kN is None:
        return None
    i_yield = int(np.argmin(np.abs(displacement_mm - dy_mm)))
    return (
        float(vy_kN),
        float(dy_mm),
        float(df["M_x_kNm"].iloc[i_yield]),
        float(df["theta_x_rad"].iloc[i_yield]),
    )


def capacity_row(
    *,
    scour_depth_m: float,
    status: str,
    n_points: int = 0,
    capacity_point: tuple[float, float, float, float] | None = None,
    target_displacement_mm: float | None = None,
    max_displacement_mm: float | None = None,
    target_reached: bool | None = None,
    raw_curve_file: Path | None = None,
) -> dict[str, float | int | str | bool | None]:
    fc_mpa, fy_mpa = nominal_materials()
    if capacity_point is None:
        vy, dy, my, theta = (None, None, None, None)
    else:
        vy, dy, my, theta = capacity_point
    return {
        "s_z_m": scour_depth_m,
        "fc_MPa": fc_mpa,
        "fy_MPa": fy_mpa,
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


def save_manifest(config: ExperimentConfig, paths: dict[str, Path]) -> None:
    target_mm, initial_incr_mm, estimated_steps = pushover_step_estimate(config)
    fc_mpa, fy_mpa = nominal_materials()
    manifest = {
        "purpose": "deterministic pushover over configured spring-removal scour depths",
        "spring_removal_scope": "all pile springs across all three pile groups/bents",
        "scour_depths_m": config.scour_depths_m,
        "terminal_note": (
            "20 m represents physical scour to the pile tip; the deepest modeled "
            "spring row is removed at 19 m, so 19 m and 20 m share the same "
            "modeled spring-removal state. The experiment does not launch "
            "OpenSees at 20 m; it records a zero-capacity terminal state."
        ),
        "nominal_materials": {"fc_mpa": fc_mpa, "fy_mpa": fy_mpa},
        "pushlimit_percent": config.pushlimit_percent,
        "target_displacement_mm": target_mm,
        "initial_increment_mm": initial_incr_mm,
        "min_increment_mm": config.min_incr_mm,
        "max_increment_mm": config.max_incr_mm,
        "displacement_control_num_iter": config.displacement_control_num_iter,
        "estimated_minimum_pushover_steps": estimated_steps,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def plot_capacity_vs_scour(capacity_df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("V_y_kN", "Yield base shear, V_y (kN)", "capacity_V_y_vs_scour.png"),
        ("Delta_y_mm", "Yield displacement, Delta_y (mm)", "capacity_Delta_y_vs_scour.png"),
        ("M_x_kNm", "Yield base moment, M_x (kNm)", "capacity_M_x_vs_scour.png"),
        ("theta_x_rad", "Yield rotation, theta_x (rad)", "capacity_theta_x_vs_scour.png"),
    ]
    output_files: list[Path] = []
    for column, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        plot_df = capacity_df.dropna(subset=[column])
        ax.plot(plot_df["s_z_m"], plot_df[column], marker="o", linewidth=1.6)
        partial_df = plot_df[plot_df["status"].astype(str).str.contains("partial")]
        if not partial_df.empty:
            ax.scatter(
                partial_df["s_z_m"],
                partial_df[column],
                marker="s",
                s=46,
                label="partial pushover",
            )
            ax.legend()
        ax.set_xlabel("Scour depth, s_z (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(",")[0] + " vs scour depth")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        output_path = plots_dir / filename
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        output_files.append(output_path)
    return output_files


def plot_overlay_curves(raw_curve_files: Iterable[Path], plots_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    curves = [load_raw_curve(path) for path in raw_curve_files if path.exists()]
    curves = [df for df in curves if not df.empty]
    if not curves:
        return []
    plots_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis", len(curves))
    output_files: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, df in enumerate(curves):
        scour_m = float(df["s_z_m"].iloc[0])
        ax.plot(df["Delta_mm"], df["V_kN"], color=cmap(idx), linewidth=1.2, label=f"{scour_m:g} m")
    ax.set_xlabel("Deck displacement, Delta (mm)")
    ax.set_ylabel("Base shear, V (kN)")
    ax.set_title("Nonlinear pushover curves by spring-removal scour depth")
    ax.grid(True, alpha=0.3)
    ax.legend(title="s_z", ncol=3, fontsize=7)
    fig.tight_layout()
    output_path = plots_dir / "V_vs_Delta_all_scour.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    output_files.append(output_path)

    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, df in enumerate(curves):
        scour_m = float(df["s_z_m"].iloc[0])
        ax.plot(df["theta_x_rad"], df["M_x_kNm"], color=cmap(idx), linewidth=1.2, label=f"{scour_m:g} m")
    ax.set_xlabel("Column rotation, theta_x (rad)")
    ax.set_ylabel("Base moment, M_x (kNm)")
    ax.set_title("Moment-rotation curves by spring-removal scour depth")
    ax.grid(True, alpha=0.3)
    ax.legend(title="s_z", ncol=3, fontsize=7)
    fig.tight_layout()
    output_path = plots_dir / "M_x_vs_theta_x_all_scour.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    output_files.append(output_path)
    return output_files


def print_plan(config: ExperimentConfig, paths: dict[str, Path]) -> None:
    target_mm, initial_incr_mm, estimated_steps = pushover_step_estimate(config)
    fc_mpa, fy_mpa = nominal_materials()
    print("Deterministic spring-removal scour pushover experiment")
    print(f"  scour depths s_z (m): {', '.join(f'{value:g}' for value in config.scour_depths_m)}")
    print("  spring removal scope: all piles in all three pile groups/bents")
    print(f"  nominal materials: fc={fc_mpa:g} MPa, fy={fy_mpa:g} MPa")
    print(f"  pushlimit: {config.pushlimit_percent:g}% of effective bridge height")
    print(
        "  pushover target: "
        f"{target_mm:g} mm in at least ~{estimated_steps} steps "
        f"(initial {initial_incr_mm:g} mm/step)"
    )
    print(f"  output root: {paths['root']}")
    print(f"  capacity tuples: {paths['capacity_tuples']}")


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    paths = output_paths(config)
    print_plan(config, paths)

    if config.dry_run:
        print("Dry run only; no OpenSeesPy model was imported or executed.")
        return 0

    ensure_output_dirs(paths)
    save_manifest(config, paths)
    rows: list[dict[str, float | int | str | bool | None]] = []
    raw_curve_files: list[Path] = []

    for scour_depth_m in config.scour_depths_m:
        csv_path = raw_curve_path(paths["raw_curves"], scour_depth_m)
        raw_curve_files.append(csv_path)
        target_mm, _, _ = pushover_step_estimate(config)

        if is_terminal_pile_tip_case(scour_depth_m):
            save_terminal_zero_curve(csv_path, scour_depth_m)
            rows.append(
                capacity_row(
                    scour_depth_m=scour_depth_m,
                    status="terminal_pile_tip_zero_capacity",
                    n_points=1,
                    capacity_point=(0.0, 0.0, 0.0, 0.0),
                    target_displacement_mm=target_mm,
                    max_displacement_mm=0.0,
                    target_reached=False,
                    raw_curve_file=csv_path,
                )
            )
            print(
                "Recorded terminal pile-tip case without OpenSees simulation: "
                f"s_z={scour_depth_m:g} m, zero capacity."
            )
            continue

        if config.skip_existing and csv_path.exists():
            df_existing = load_raw_curve(csv_path)
            max_disp = float(df_existing["Delta_mm"].max()) if "Delta_mm" in df_existing else None
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
                    scour_depth_m=scour_depth_m,
                    status=status,
                    n_points=len(df_existing),
                    capacity_point=capacity_from_raw_curve(df_existing),
                    target_displacement_mm=target_mm,
                    max_displacement_mm=max_disp,
                    target_reached=target_reached,
                    raw_curve_file=csv_path,
                )
            )
            print(f"Skipping existing s_z={scour_depth_m:g} m: {csv_path}")
            continue

        print(f"\nRunning nonlinear pushover for s_z={scour_depth_m:g} m...")
        try:
            curve = run_one_depth(config, scour_depth_m)
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"Required runtime dependency is missing: {exc}")
            return 2

        if curve is None:
            rows.append(
                capacity_row(
                    scour_depth_m=scour_depth_m,
                    status="failed",
                    target_displacement_mm=target_mm,
                    target_reached=False,
                )
            )
            print(f"Simulation failed for s_z={scour_depth_m:g} m.")
            continue

        save_raw_curve(curve, csv_path)
        max_disp = float(max(curve.displacement_mm))
        target_reached = max_disp >= target_mm - 1.0e-6
        status = "ok" if target_reached else "partial"
        rows.append(
            capacity_row(
                scour_depth_m=scour_depth_m,
                status=status,
                n_points=len(curve.displacement_mm),
                capacity_point=curve.capacity_point,
                target_displacement_mm=target_mm,
                max_displacement_mm=max_disp,
                target_reached=target_reached,
                raw_curve_file=csv_path,
            )
        )
        print(f"Saved {'raw' if target_reached else 'partial raw'} curve: {csv_path}")

    capacity_df = pd.DataFrame(rows)
    capacity_df.to_csv(paths["capacity_tuples"], index=False)
    print(f"\nSaved capacity tuples: {paths['capacity_tuples']}")

    plot_files = plot_capacity_vs_scour(capacity_df, paths["plots"])
    plot_files.extend(plot_overlay_curves(raw_curve_files, paths["plots"]))
    for plot_file in plot_files:
        print(f"Saved plot: {plot_file}")

    if capacity_df.empty or (capacity_df["status"] == "failed").all():
        print("No successful or partial simulations completed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
