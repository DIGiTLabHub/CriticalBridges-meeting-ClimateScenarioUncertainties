#!/usr/bin/env python
"""Scenario pushover runner for manuscript-style scour cases.

This entry point runs two explicit scour depths per climate scenario:
mean - 2 standard deviations and mean + 2 standard deviations from the
scenario-specific LHS scour hazard model. Each case runs the OpenSees
pushover model, extracts the capacity tuple, and saves a pushover plot with
the bilinear fit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.parameters import GEOMETRY, MATERIALS, SCOUR
from src.postprocessing.bilinear_fit import fit_bilinear_profile
from src.scour import LHS_scour_hazard


DEFAULT_SCENARIOS = ("missouri", "colorado", "extreme")


@dataclass(frozen=True)
class PushoverCase:
    scenario: str
    label: str
    scour_depth_m: float
    scour_mean_m: float
    scour_std_m: float
    fc_MPa: float
    fy_MPa: float


@dataclass(frozen=True)
class PushoverResult:
    case: PushoverCase
    capacity_point: tuple[float, float, float, float]
    displacement_mm: np.ndarray
    base_shear_kN: np.ndarray
    rotation_rad: np.ndarray
    base_moment_kNm: np.ndarray
    k1_kN_per_mm: float
    k2_kN_per_mm: float


def scenario_scour_statistics(
    scenario: str,
    *,
    samples: int,
    seed: int | None,
) -> tuple[float, float]:
    params = SCOUR["scenarios"][scenario]
    hazard = LHS_scour_hazard(
        lhsN=samples,
        vel=params["velocity_m_s"],
        dPier=GEOMETRY["pier_diameter_m"],
        gama=SCOUR["kinematic_viscosity_m2_s"],
        zDot=params["erosion_rate_mm_hr"],
        random_seed=seed,
    )
    return float(hazard["z50Mean"]), float(hazard["z50std"])


def build_scenario_cases(
    scenarios: Iterable[str],
    *,
    hazard_samples: int = 1000,
    seed: int | None = 42,
    sample_materials: bool = False,
) -> list[PushoverCase]:
    cases: list[PushoverCase] = []
    fc_mean = float(MATERIALS["concrete"]["mean_MPa"])
    fy_mean = float(MATERIALS["steel"]["mean_MPa"])
    rng = np.random.default_rng(seed)

    for index, scenario in enumerate(scenarios):
        mean_m, std_m = scenario_scour_statistics(
            scenario,
            samples=hazard_samples,
            seed=None if seed is None else seed + index,
        )
        depth_specs = (
            ("avg_minus_2std", mean_m - 2.0 * std_m),
            ("avg_plus_2std", mean_m + 2.0 * std_m),
        )

        for label, depth_m in depth_specs:
            # A negative scour depth is nonphysical; keep a small positive value
            # so log/inverse surrogate features remain defined.
            scour_depth_m = max(float(depth_m), 1.0e-3)
            if sample_materials:
                fc_MPa = float(
                    rng.normal(
                        MATERIALS["concrete"]["mean_MPa"],
                        MATERIALS["concrete"]["std_MPa"],
                    )
                )
                fy_MPa = float(
                    rng.lognormal(
                        np.log(MATERIALS["steel"]["mean_MPa"]),
                        MATERIALS["steel"]["std_MPa"] / MATERIALS["steel"]["mean_MPa"],
                    )
                )
            else:
                fc_MPa = fc_mean
                fy_MPa = fy_mean

            cases.append(
                PushoverCase(
                    scenario=scenario,
                    label=label,
                    scour_depth_m=scour_depth_m,
                    scour_mean_m=mean_m,
                    scour_std_m=std_m,
                    fc_MPa=fc_MPa,
                    fy_MPa=fy_MPa,
                )
            )

    return cases


def run_case(case: PushoverCase, *, seed: int | None = None) -> PushoverResult | None:
    from BridgeModeling.Pushover import run_single_pushover_simulation

    result = run_single_pushover_simulation(
        scenario=case.scenario,
        random_seed=seed,
        scour_depth_m=case.scour_depth_m,
        fc_MPa=case.fc_MPa,
        fy_MPa=case.fy_MPa,
    )
    if result is None:
        return None

    capacity_point, displacement_mm, base_shear_kN, rotation_rad, base_moment_kNm = result
    k1, k2, _, _ = fit_bilinear_profile(displacement_mm, base_shear_kN)

    return PushoverResult(
        case=case,
        capacity_point=capacity_point,
        displacement_mm=displacement_mm,
        base_shear_kN=base_shear_kN,
        rotation_rad=rotation_rad,
        base_moment_kNm=base_moment_kNm,
        k1_kN_per_mm=float(k1),
        k2_kN_per_mm=float(k2),
    )


def save_case_plot(result: PushoverResult, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    case = result.case
    vy, dy, my, thy = result.capacity_point
    d_range = np.linspace(result.displacement_mm.min(), result.displacement_mm.max(), 500)
    v_model = np.where(
        d_range < dy,
        result.k1_kN_per_mm * d_range,
        result.k2_kN_per_mm * d_range
        + (result.k1_kN_per_mm - result.k2_kN_per_mm) * dy,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(result.displacement_mm, result.base_shear_kN, s=12, alpha=0.45)
    axes[0].plot(d_range, v_model, color="black", linewidth=2)
    axes[0].scatter([dy], [vy], color="red", zorder=5)
    axes[0].set_xlabel("Deck displacement, Dy (mm)")
    axes[0].set_ylabel("Base shear, Vy (kN)")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(result.rotation_rad, result.base_moment_kNm, s=12, alpha=0.45)
    axes[1].scatter([thy], [my], color="red", zorder=5)
    axes[1].set_xlabel("Column rotation, Thy (rad)")
    axes[1].set_ylabel("Base moment, My (kNm)")
    axes[1].grid(True, alpha=0.3)

    tuple_text = (
        f"Scour = {case.scour_depth_m:.3f} m\n"
        f"Vy = {vy:.2f} kN\n"
        f"Dy = {dy:.2f} mm\n"
        f"My = {my:.2f} kNm\n"
        f"Thy = {thy:.5f} rad"
    )
    axes[0].text(
        0.03,
        0.97,
        tuple_text,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.65", "alpha": 0.9},
        fontsize=9,
    )

    title = f"{case.scenario} | {case.label} | mean={case.scour_mean_m:.3f} m, std={case.scour_std_m:.3f} m"
    fig.suptitle(title)
    fig.tight_layout()

    scenario_dir = output_dir / case.scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    plot_path = scenario_dir / f"{case.label}_scour_{case.scour_depth_m * 1000:.1f}mm.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path


def results_to_workbook(results: list[PushoverResult], output_excel: Path) -> Path:
    rows_by_scenario: dict[str, list[dict[str, float | str]]] = {}

    for result in results:
        case = result.case
        vy, dy, my, thy = result.capacity_point
        rows_by_scenario.setdefault(case.scenario, []).append(
            {
                "Case": case.label,
                "Scour_Depth_mm": case.scour_depth_m * 1000.0,
                "Scour_Mean_m": case.scour_mean_m,
                "Scour_Std_m": case.scour_std_m,
                "fc_MPa": case.fc_MPa,
                "fy_MPa": case.fy_MPa,
                "Vy_kN": vy,
                "dy_mm": dy,
                "My_kNm": my,
                "Thy_rad": thy,
                "k1_kN_per_mm": result.k1_kN_per_mm,
                "k2_kN_per_mm": result.k2_kN_per_mm,
            }
        )

    output_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_excel) as writer:
        for scenario, rows in rows_by_scenario.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=scenario, index=False)
    return output_excel


def train_surrogates_from_workbook(
    output_excel: Path,
    *,
    base_results_folder: Path,
    n_credal: int,
) -> None:
    from src.surrogate_modeling.training import run_training_workflow

    run_training_workflow(
        excel_path=output_excel,
        base_results_folder=base_results_folder,
        model_types=("svr", "gbr"),
        n_credal=n_credal,
        test_size=0.34,
        split_seed=42,
        apply_iqr_filter=False,
        save_plots=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run explicit avg±2std scour pushover cases for all climate scenarios, "
            "plot bilinear capacity extraction, and optionally train surrogates."
        )
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        choices=list(DEFAULT_SCENARIOS),
        help="Scenario names to run.",
    )
    parser.add_argument(
        "--hazard-samples",
        type=int,
        default=1000,
        help="LHS samples used to estimate scenario mean/std scour depths.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--sample-materials",
        action="store_true",
        help="Sample fc/fy per case instead of using material means.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("Plots") / "scenario_pushover_cases",
        help="Directory for pushover/bilinear plots.",
    )
    parser.add_argument(
        "--output-excel",
        type=Path,
        default=Path("RecorderData") / "Yield_Results_selected_scenarios.xlsx",
        help="Workbook path for extracted capacity tuples.",
    )
    parser.add_argument(
        "--train-surrogates",
        action="store_true",
        help="Train SVR/GBR surrogate ensembles from the generated workbook.",
    )
    parser.add_argument(
        "--surrogate-results-dir",
        type=Path,
        default=Path("RecorderData") / "results_selected_cases",
        help="Output folder for optional surrogate training artifacts.",
    )
    parser.add_argument(
        "--n-credal",
        type=int,
        default=30,
        help="Bootstrap models per target when --train-surrogates is used.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = build_scenario_cases(
        args.scenarios,
        hazard_samples=args.hazard_samples,
        seed=args.seed,
        sample_materials=args.sample_materials,
    )

    print("Selected pushover cases:")
    for case in cases:
        print(
            f"  {case.scenario:8s} {case.label:14s} "
            f"scour={case.scour_depth_m:.3f} m "
            f"(mean={case.scour_mean_m:.3f}, std={case.scour_std_m:.3f})"
        )

    results: list[PushoverResult] = []
    for index, case in enumerate(cases):
        print(f"\nRunning {case.scenario}/{case.label}...")
        result = run_case(case, seed=None if args.seed is None else args.seed + index)
        if result is None:
            print(f"Skipped {case.scenario}/{case.label}: simulation failed.")
            continue
        results.append(result)
        plot_path = save_case_plot(result, args.plot_dir)
        print(f"Saved plot: {plot_path}")

    if not results:
        print("No successful pushover simulations; no workbook or surrogate models created.")
        return 1

    workbook_path = results_to_workbook(results, args.output_excel)
    print(f"\nSaved capacity workbook: {workbook_path}")

    if args.train_surrogates:
        if len(results) < 6:
            print("Surrogate training skipped: fewer than 6 successful cases.")
        else:
            print("Training surrogate ensembles from selected cases...")
            train_surrogates_from_workbook(
                workbook_path,
                base_results_folder=args.surrogate_results_dir,
                n_credal=args.n_credal,
            )
            print(f"Saved surrogate artifacts under: {args.surrogate_results_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
