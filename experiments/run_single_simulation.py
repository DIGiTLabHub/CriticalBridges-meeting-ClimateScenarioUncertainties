#!/usr/bin/env python
"""Run one stochastic scenario sample through nonlinear pushover extraction.

This experiment samples a full stochastic input table for one flooding scenario,
selects one row, runs the OpenSeesPy bridge pushover model for that sampled
scour/material state, and saves the raw curve plus the bilinear capacity tuple.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from BridgeModeling.Pushover import run_single_pushover_simulation
from BridgeModeling.ZeroLengthElement import effective_scour_depth_mm
from config.parameters import ANALYSIS, MATERIALS, SCOUR
from src.postprocessing.bilinear_fit import fit_bilinear_profile
from src.scour import LHS_scour_hazard


DEFAULT_OUTPUT_DIR = Path("experiments") / "output" / "single_simulation"


@dataclass(frozen=True)
class SamplePoint:
    sample_index: int
    scenario: str
    scour_continuous_m: float
    scour_model_m: float
    fc_mpa: float
    fy_mpa: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample one stochastic scour/material input row for a flooding "
            "scenario, run nonlinear pushover, and extract the capacity tuple."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCOUR["scenarios"].keys()),
        default="missouri",
        help="Flooding/scour scenario used for LHS scour sampling.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of stochastic rows to generate before selecting one sample.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based row index selected from the generated stochastic table.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--continuous-scour",
        action="store_true",
        help=(
            "Run the raw continuous scour depth. By default the script passes "
            "the equivalent discrete spring-removal depth to the FE model."
        ),
    )
    parser.add_argument(
        "--pushlimit",
        type=float,
        default=None,
        help="Override pushover drift limit as percent of effective bridge height.",
    )
    parser.add_argument(
        "--initial-incr-mm",
        type=float,
        default=None,
        help="Override initial DisplacementControl increment in mm.",
    )
    parser.add_argument(
        "--min-incr-mm",
        type=float,
        default=None,
        help="Override minimum adaptive DisplacementControl increment in mm.",
    )
    parser.add_argument(
        "--max-incr-mm",
        type=float,
        default=None,
        help="Override maximum adaptive DisplacementControl increment in mm.",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=None,
        help="Override desired iteration count for adaptive DisplacementControl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the selected sample, raw curve, summary, and plots.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise SystemExit("--samples must be positive.")
    if args.sample_index < 0 or args.sample_index >= args.samples:
        raise SystemExit("--sample-index must satisfy 0 <= index < --samples.")
    if args.pushlimit is not None and args.pushlimit <= 0:
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


def apply_pushover_overrides(args: argparse.Namespace) -> None:
    pushover = ANALYSIS["pushover"]
    if args.pushlimit is not None:
        pushover["max_drift_ratio"] = args.pushlimit / 100.0
    if args.initial_incr_mm is not None:
        pushover["displacement_increment_mm"] = args.initial_incr_mm
    if args.min_incr_mm is not None:
        pushover["displacement_increment_min_mm"] = args.min_incr_mm
    if args.max_incr_mm is not None:
        pushover["displacement_increment_max_mm"] = args.max_incr_mm
    if args.num_iter is not None:
        pushover["displacement_control_num_iter"] = args.num_iter


def sample_stochastic_table(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(args.seed)
    scenario_params = SCOUR["scenarios"][args.scenario]

    hazard = LHS_scour_hazard(
        lhsN=args.samples,
        vel=scenario_params["velocity_m_s"],
        dPier=1.5,
        gama=SCOUR["kinematic_viscosity_m2_s"],
        zDot=scenario_params["erosion_rate_mm_hr"],
        rng=rng,
    )

    fc_samples = rng.normal(
        MATERIALS["concrete"]["mean_MPa"],
        MATERIALS["concrete"]["std_MPa"],
        args.samples,
    )
    fy_samples = rng.lognormal(
        np.log(MATERIALS["steel"]["mean_MPa"]),
        MATERIALS["steel"]["std_MPa"] / MATERIALS["steel"]["mean_MPa"],
        args.samples,
    )

    table = pd.DataFrame(
        {
            "sample_index": np.arange(args.samples, dtype=int),
            "scenario": args.scenario,
            "s_z_continuous_m": hazard["z50Final"],
            "fc_MPa": fc_samples,
            "fy_MPa": fy_samples,
        }
    )
    table["s_z_model_m"] = table["s_z_continuous_m"].apply(
        lambda value: effective_scour_depth_mm(float(value) * 1000.0) / 1000.0
    )
    return table, hazard


def select_sample(table: pd.DataFrame, args: argparse.Namespace) -> SamplePoint:
    row = table.iloc[int(args.sample_index)]
    scour_model_m = (
        float(row["s_z_continuous_m"])
        if args.continuous_scour
        else float(row["s_z_model_m"])
    )
    return SamplePoint(
        sample_index=int(row["sample_index"]),
        scenario=str(row["scenario"]),
        scour_continuous_m=float(row["s_z_continuous_m"]),
        scour_model_m=scour_model_m,
        fc_mpa=float(row["fc_MPa"]),
        fy_mpa=float(row["fy_MPa"]),
    )


def output_paths(args: argparse.Namespace, sample: SamplePoint) -> dict[str, Path]:
    sample_dir = (
        args.output_dir
        / sample.scenario
        / f"sample_{sample.sample_index:04d}_scour_{sample.scour_model_m:.3f}m"
    )
    return {
        "sample_dir": sample_dir,
        "sample_table": sample_dir / "stochastic_samples.csv",
        "summary": sample_dir / "summary.json",
        "raw_curve": sample_dir / "raw_pushover_curve.csv",
        "dv_plot": sample_dir / "pushover_Delta_V.png",
        "mt_plot": sample_dir / "pushover_theta_M.png",
    }


def save_raw_curve(
    path: Path,
    sample: SamplePoint,
    displacement_mm: np.ndarray,
    base_shear_kN: np.ndarray,
    rotation_rad: np.ndarray,
    base_moment_kNm: np.ndarray,
) -> None:
    min_len = min(
        len(displacement_mm), len(base_shear_kN), len(rotation_rad), len(base_moment_kNm)
    )
    df = pd.DataFrame(
        {
            "sample_index": sample.sample_index,
            "scenario": sample.scenario,
            "s_z_continuous_m": sample.scour_continuous_m,
            "s_z_model_m": sample.scour_model_m,
            "fc_MPa": sample.fc_mpa,
            "fy_MPa": sample.fy_mpa,
            "Delta_mm": displacement_mm[:min_len],
            "V_kN": base_shear_kN[:min_len],
            "theta_x_rad": rotation_rad[:min_len],
            "M_x_kNm": base_moment_kNm[:min_len],
        }
    )
    df.to_csv(path, index=False)


def save_plots(
    paths: dict[str, Path],
    sample: SamplePoint,
    capacity_point: tuple[float, float, float, float],
    displacement_mm: np.ndarray,
    base_shear_kN: np.ndarray,
    rotation_rad: np.ndarray,
    base_moment_kNm: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    vy, dy, my, thy = capacity_point
    k1, k2, dy_fit, _ = fit_bilinear_profile(displacement_mm, base_shear_kN)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(displacement_mm, base_shear_kN, linewidth=1.5, label="Pushover curve")
    if k1 is not None and k2 is not None and dy_fit is not None:
        d_range = np.linspace(displacement_mm.min(), displacement_mm.max(), 500)
        v_model = np.where(
            d_range < dy_fit,
            k1 * d_range,
            k2 * d_range + (k1 - k2) * dy_fit,
        )
        ax.plot(d_range, v_model, "--", linewidth=1.8, label="Bilinear fit")
    ax.plot([dy], [vy], "o", markersize=6, label="Capacity tuple point")
    ax.set_xlabel("Deck displacement, Delta (mm)")
    ax.set_ylabel("Base shear, V (kN)")
    ax.set_title(f"{sample.scenario} sample {sample.sample_index}: V-Delta")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["dv_plot"], dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(rotation_rad, base_moment_kNm, linewidth=1.5, label="Pushover curve")
    ax.plot([thy], [my], "o", markersize=6, label="Capacity tuple point")
    ax.set_xlabel("Column rotation, theta_x (rad)")
    ax.set_ylabel("Base moment, M_x (kNm)")
    ax.set_title(f"{sample.scenario} sample {sample.sample_index}: M-theta")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["mt_plot"], dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    apply_pushover_overrides(args)

    table, hazard = sample_stochastic_table(args)
    sample = select_sample(table, args)
    paths = output_paths(args, sample)
    paths["sample_dir"].mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["sample_table"], index=False)

    print("Single stochastic nonlinear pushover simulation")
    print(f"  scenario: {sample.scenario}")
    print(f"  generated samples: {args.samples}")
    print(f"  selected sample index: {sample.sample_index}")
    print(f"  continuous s_z: {sample.scour_continuous_m:.4f} m")
    print(f"  model s_z: {sample.scour_model_m:.4f} m")
    print(f"  fc: {sample.fc_mpa:.3f} MPa")
    print(f"  fy: {sample.fy_mpa:.3f} MPa")
    print(f"  hazard mean/std: {hazard['z50Mean']:.4f}/{hazard['z50std']:.4f} m")
    print(f"  output: {paths['sample_dir']}")

    result = run_single_pushover_simulation(
        scenario=sample.scenario,
        random_seed=args.seed,
        scour_depth_m=sample.scour_model_m,
        fc_MPa=sample.fc_mpa,
        fy_MPa=sample.fy_mpa,
    )

    if result is None:
        summary = {
            "status": "failed",
            "sample": sample.__dict__,
            "hazard_mean_m": float(hazard["z50Mean"]),
            "hazard_std_m": float(hazard["z50std"]),
        }
        paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
        print("Simulation failed.")
        return 1

    capacity_point, displacement_mm, base_shear_kN, rotation_rad, base_moment_kNm = result
    vy, dy, my, thy = (float(value) for value in capacity_point)
    save_raw_curve(
        paths["raw_curve"],
        sample,
        displacement_mm,
        base_shear_kN,
        rotation_rad,
        base_moment_kNm,
    )

    if not args.no_plots:
        save_plots(
            paths,
            sample,
            capacity_point,
            displacement_mm,
            base_shear_kN,
            rotation_rad,
            base_moment_kNm,
        )

    summary = {
        "status": "ok",
        "sample": sample.__dict__,
        "hazard_mean_m": float(hazard["z50Mean"]),
        "hazard_std_m": float(hazard["z50std"]),
        "capacity_tuple": {
            "V_y_kN": vy,
            "Delta_y_mm": dy,
            "M_x_kNm": my,
            "theta_x_rad": thy,
        },
        "n_curve_points": int(len(displacement_mm)),
        "raw_curve": str(paths["raw_curve"]),
        "plots": [] if args.no_plots else [str(paths["dv_plot"]), str(paths["mt_plot"])],
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")

    print("Capacity tuple extracted:")
    print(f"  V_y: {vy:.3f} kN")
    print(f"  Delta_y: {dy:.3f} mm")
    print(f"  M_x: {my:.3f} kNm")
    print(f"  theta_x: {thy:.6g} rad")
    print(f"Saved raw curve: {paths['raw_curve']}")
    print(f"Saved summary: {paths['summary']}")
    if not args.no_plots:
        print(f"Saved plots: {paths['dv_plot']}, {paths['mt_plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
