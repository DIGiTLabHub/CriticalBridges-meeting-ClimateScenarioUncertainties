#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surrogate Modeling Testing Script

Tests surrogate models for a specific climate scenario.
Reads spreadsheet results, trains ML models (GBR, SVR), computes credal sets,
analyzes capacity trajectories, and generates distribution plots.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100


def load_scenario_data(scenario, excel_path="RecorderData/Yield_Results_by_Scenario.xlsx"):
    """Load and filter data for a specific scenario."""
    try:
        df = pd.read_excel(excel_path, sheet_name=scenario)
    except KeyError:
        print(f"❌ Scenario '{scenario}' not found in spreadsheet")
        return None

    df = df[
        (df["Scour_Depth_mm"] > 0) &
        (df["Vy_kN"] > 0) &
        (df["dy_mm"] > 0) &
        (df["My_kNm"] > 0) &
        (df["Thy_rad"] > 0)
    ].copy()
    df.rename(columns={
        "Scour_Depth_mm": "Scour",
        "Vy_kN": "Vy",
        "dy_mm": "Dy",
        "My_kNm": "My",
        "Thy_rad": "Thy"
    }, inplace=True)

    print(f"✅ Loaded {len(df)} samples for scenario '{scenario}'")
    return df


def create_features(df):
    """Create polynomial features for ML models."""
    df = df.copy()
    df["Scour2"] = df["Scour"] ** 2
    df["Scour3"] = df["Scour"] ** 3
    df["invScour"] = 1 / df["Scour"]
    df["logScour"] = np.log(df["Scour"])
    df["sqrtScour"] = np.sqrt(df["Scour"])

    features = ["Scour", "Scour2", "Scour3", "logScour", "invScour", "sqrtScour"]
    return df, features


def train_model_bootstrap(X_train, y_train, model_type, n_bootstrap=30, random_state=42):
    """Train bootstrap ensemble for credal sets."""
    models = []

    for i in range(n_bootstrap):
        X_res, y_res = resample(X_train, y_train, replace=True, random_state=random_state + i)

        if model_type == "gbr":
            model = GradientBoostingRegressor(
                n_estimators=700,
                max_depth=3,
                learning_rate=0.015,
                subsample=0.85,
                random_state=random_state + i
            )
        elif model_type == "svr":
            model = SVR(
                kernel='rbf',
                C=100.0,
                epsilon=0.01,
                gamma='scale'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_res, y_res)
        models.append(model)

    return models


def compute_credal_bounds(models, X_test, y_scaler=None):
    """Compute credal bounds from ensemble predictions."""
    predictions = []

    for model in models:
        if hasattr(model, 'predict'):
            pred = model.predict(X_test)
            if y_scaler is not None:
                pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
            predictions.append(pred)

    predictions = np.vstack(predictions)
    lower = np.min(predictions, axis=0)
    upper = np.max(predictions, axis=0)
    median = np.median(predictions, axis=0)

    return lower, upper, median


def plot_capacity_trajectory(df, credal_bounds, target_pairs, output_dir, scenario):
    """Plot capacity trajectories with credal bounds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (x_target, y_target) in enumerate(target_pairs):
        ax = axes[idx]

        ax.scatter(df[x_target], df[y_target], alpha=0.6, s=30, label="Data Points")

        if credal_bounds and x_target in credal_bounds and y_target in credal_bounds:
            x_lower, x_upper, x_median = credal_bounds[x_target]
            y_lower, y_upper, y_median = credal_bounds[y_target]

            ax.plot(x_median, y_median, 'r-', linewidth=2, label="Median Trajectory")
            ax.fill_betweenx(y_median, x_lower, x_upper, alpha=0.3, color='red', label="Credal Bounds")

        ax.set_xlabel(f"{x_target} ({'kN' if x_target=='Vy' else 'mm' if x_target=='Dy' else 'kNm' if x_target=='My' else 'rad'})")
        ax.set_ylabel(f"{y_target} ({'kN' if y_target=='Vy' else 'mm' if y_target=='Dy' else 'kNm' if y_target=='My' else 'rad'})")
        ax.set_title(f"{x_target} vs {y_target} Trajectory")
        ax.legend()
        ax.grid(True)

    plt.suptitle(f"Capacity Trajectories - Scenario: {scenario}")
    plt.tight_layout()
    plt.savefig(output_dir / f"capacity_trajectories_{scenario}.png", dpi=300, bbox_inches='tight')
    plt.close()


def select_critical_locations(df, n_points=3):
    """Select critical locations for detailed analysis."""
    critical_points = []

    df_sorted = df.sort_values('Vy')
    quantiles = np.linspace(0.1, 0.9, n_points)
    for q in quantiles:
        idx = int(q * (len(df_sorted) - 1))
        point = df_sorted.iloc[idx]
        critical_points.append({
            'scour': point['Scour'],
            'Vy': point['Vy'],
            'Dy': point['Dy'],
            'My': point['My'],
            'Thy': point['Thy'],
            'quantile': q
        })

    return critical_points


def plot_distribution_histograms(df, critical_points, output_dir, scenario):
    """Create histograms of capacity distributions at critical locations."""
    variables = ['Vy', 'Dy', 'My', 'Thy']
    units = {'Vy': 'kN', 'Dy': 'mm', 'My': 'kNm', 'Thy': 'rad'}

    for point in critical_points:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, var in enumerate(variables):
            ax = axes[idx]

            ax.hist(df[var], bins=30, alpha=0.7, color='blue', label='All Data')
            ax.axvline(point[var], color='red', linestyle='--', linewidth=2,
                      label=f'Critical Point: {point[var]:.1f} {units[var]}')

            ax.set_xlabel(f'{var} ({units[var]})')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{var} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('.1f')
        plt.tight_layout()
        plt.savefig(output_dir / f'distributions_scour_{point["scour"]:.0f}mm_{scenario}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test surrogate models for bridge capacity prediction"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["missouri", "colorado", "extreme"],
        help="Climate scenario to analyze"
    )
    parser.add_argument(
        "--methods",
        default="gbr,svr",
        help="ML methods to use (comma-separated: gbr,svr)"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=30,
        help="Number of bootstrap samples for credal sets"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate distribution plots"
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/test_output",
        help="Output directory for results"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) / f"{args.scenario}_surrogate_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🏃 Testing surrogate models for scenario: {args.scenario}")
    print(f"📁 Output directory: {output_dir}")

    df = load_scenario_data(args.scenario)
    if df is None:
        return

    df, features = create_features(df)

    targets = ["Vy", "Dy", "My", "Thy"]
    X = df[features].values
    X_train, X_test, y_train_full, y_test_full = train_test_split(
        X, df[targets].values, test_size=0.15, random_state=42
    )

    methods = [m.strip() for m in args.methods.split(',')]
    credal_bounds = {}
    model_performance = []

    for target_idx, target in enumerate(targets):
        print(f"\n🔄 Training models for {target}...")

        y_train = y_train_full[:, target_idx]
        y_test = y_test_full[:, target_idx]

        for method in methods:
            print(f"  📊 Method: {method.upper()}")

        models = train_model_bootstrap(X_train, y_train, method, args.bootstrap)
        lower, upper, median = compute_credal_bounds(models, X_test)
        credal_bounds[f"{target}_{method}"] = (lower, upper, median)
            r2 = r2_score(y_test, median)
            rmse = mean_squared_error(y_test, median, squared=False)

            model_performance.append({
                'Target': target,
                'Method': method.upper(),
                'R2': r2,
                'RMSE': rmse,
                'Credal_Width_Mean': np.mean(upper - lower)
            })

            print(f"    ✅ R² = {r2:.3f}, RMSE = {rmse:.3f}, Credal Width = {np.mean(upper - lower):.3f}")

            for i, model in enumerate(models):
                joblib.dump(model, output_dir / f"{target}_{method}_boot{i}.pkl")

    perf_df = pd.DataFrame(model_performance)
    perf_df.to_excel(output_dir / f"model_performance_{args.scenario}.xlsx", index=False)

    print(f"\n📊 Model Performance Summary:")
    print(perf_df.to_string(index=False))

    critical_points = select_critical_locations(df)
    print(f"\n🎯 Selected {len(critical_points)} critical locations for analysis")

    if args.plots:
        print("📈 Generating plots...")
        plot_capacity_trajectory(df, None, [("Vy", "Dy"), ("My", "Thy")], output_dir, args.scenario)
        plot_distribution_histograms(df, critical_points, output_dir, args.scenario)
        print(f"✅ Plots saved to {output_dir}")

    critical_df = pd.DataFrame(critical_points)
    critical_df.to_excel(output_dir / f"critical_points_{args.scenario}.xlsx", index=False)

    print(f"\n✅ Analysis complete! Results saved to {output_dir}")
    print("📁 Files generated:")
    print(f"  - model_performance_{args.scenario}.xlsx")
    print(f"  - critical_points_{args.scenario}.xlsx")
    if args.plots:
        print("  - capacity_trajectories_{args.scenario}.png"        print("  - distribution histograms for critical points"


if __name__ == "__main__":
    main()