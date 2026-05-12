from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_EXCEL_PATH = Path("RecorderData/Yield_Results_by_Scenario.xlsx")
DEFAULT_BASE_RESULTS = Path("RecorderData/results")
DEFAULT_TUPLE_RESULTS = DEFAULT_BASE_RESULTS / "Tuple_Data_Process"
DEFAULT_SVR_FOLDER = DEFAULT_TUPLE_RESULTS / "ML_Surrogate_Credal_SVR"
DEFAULT_GBR_FOLDER = DEFAULT_TUPLE_RESULTS / "ML_Surrogate_Credal"

FEATURES = ["Scour", "Scour2", "Scour3", "logScour", "invScour", "sqrtScour"]
TARGETS = ["Vy", "Dy", "My", "Thy"]

RAW_TO_MODEL_COLUMNS = {
    "Scour_Depth_mm": "Scour",
    "Vy_kN": "Vy",
    "dy_mm": "Dy",
    "My_kNm": "My",
    "Thy_rad": "Thy",
}


def _import_ml_deps():
    import joblib
    import matplotlib.pyplot as plt
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.utils import resample

    return {
        "joblib": joblib,
        "plt": plt,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "train_test_split": train_test_split,
        "StandardScaler": StandardScaler,
        "SVR": SVR,
        "resample": resample,
    }


def load_yield_results(excel_path: Path = DEFAULT_EXCEL_PATH) -> pd.DataFrame:
    return pd.concat(pd.read_excel(excel_path, sheet_name=None), ignore_index=True)


def clean_yield_results(df_all: pd.DataFrame) -> pd.DataFrame:
    required_cols = list(RAW_TO_MODEL_COLUMNS.keys())
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise KeyError(f"Missing required columns in yield results: {missing}")

    df = df_all[
        (df_all["Scour_Depth_mm"] > 0)
        & (df_all["Vy_kN"] > 0)
        & (df_all["dy_mm"] > 0)
        & (df_all["My_kNm"] > 0)
        & (df_all["Thy_rad"] > 0)
    ].copy()

    df.rename(columns=RAW_TO_MODEL_COLUMNS, inplace=True)
    return df


def filter_iqr_all_tuples(df: pd.DataFrame, bin_width: float = 250.0) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    max_scour = float(df["Scour"].max())
    bins = np.arange(0.0, max_scour + bin_width, bin_width)
    if bins.size < 2:
        bins = np.array([0.0, max_scour + bin_width], dtype=float)

    labels = (bins[:-1] + bins[1:]) / 2.0
    grouped_df = df.copy()
    grouped_df["Scour_Bin"] = pd.cut(grouped_df["Scour"], bins=bins, labels=labels)

    def _filter_group(group: pd.DataFrame) -> pd.DataFrame:
        mask = np.ones(len(group), dtype=bool)
        for col in TARGETS:
            q1 = group[col].quantile(0.25)
            q3 = group[col].quantile(0.75)
            mask &= (group[col] >= q1) & (group[col] <= q3)
        return group[mask]

    filtered = grouped_df.groupby("Scour_Bin", group_keys=False).apply(_filter_group)
    return filtered.drop(columns=["Scour_Bin"], errors="ignore")


def engineer_scour_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Scour2"] = out["Scour"] ** 2
    out["Scour3"] = out["Scour"] ** 3
    out["invScour"] = 1.0 / out["Scour"]
    out["logScour"] = np.log(out["Scour"])
    out["sqrtScour"] = np.sqrt(out["Scour"])
    return out


def prepare_training_dataframe(
    excel_path: Path = DEFAULT_EXCEL_PATH,
    *,
    apply_iqr_filter: bool = True,
    bin_width: float = 250.0,
) -> pd.DataFrame:
    df_all = load_yield_results(excel_path)
    df_clean = clean_yield_results(df_all)
    df_model = (
        filter_iqr_all_tuples(df_clean, bin_width=bin_width)
        if apply_iqr_filter
        else df_clean
    )
    return engineer_scour_features(df_model)


def _save_credal_plot(
    plt,
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    y_median: np.ndarray,
    y_all: np.ndarray,
    target: str,
    model_label: str,
    plot_path: Path,
) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(
        y_true, y_median, color="blue", s=30, alpha=0.6, label="Median Prediction"
    )
    plt.fill_between(
        np.arange(len(y_true)),
        y_lower,
        y_upper,
        alpha=0.3,
        label="Credal Bounds [min–max]",
    )
    plt.plot(
        [y_all.min(), y_all.max()], [y_all.min(), y_all.max()], "r--", label="1:1 Line"
    )
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"{model_label} Credal Bounds for {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def train_svr_credal_ensemble(
    df_features: pd.DataFrame,
    *,
    output_folder: Path = DEFAULT_SVR_FOLDER,
    n_credal: int = 30,
    test_size: float = 0.15,
    split_seed: int = 42,
    save_plots: bool = True,
) -> pd.DataFrame:
    deps = _import_ml_deps()
    joblib = deps["joblib"]
    plt = deps["plt"]
    train_test_split = deps["train_test_split"]
    StandardScaler = deps["StandardScaler"]
    SVR = deps["SVR"]
    resample = deps["resample"]
    r2_score = deps["r2_score"]
    mean_squared_error = deps["mean_squared_error"]

    output_folder.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, float | str]] = []

    for target in TARGETS:
        X = df_features[FEATURES].values
        y = df_features[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed
        )

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

        joblib.dump(x_scaler, output_folder / f"x_scaler_{target}.pkl")
        joblib.dump(y_scaler, output_folder / f"y_scaler_{target}.pkl")

        y_preds: list[np.ndarray] = []
        for i in range(n_credal):
            X_res, y_res = resample(
                X_train_scaled, y_train_scaled, replace=True, random_state=i
            )
            model = SVR(
                kernel="rbf",
                C=100.0,
                epsilon=0.01,
                gamma="scale",
                shrinking=True,
                tol=1e-4,
                cache_size=500,
            )
            model.fit(X_res, y_res)
            joblib.dump(model, output_folder / f"credal_svr_model_{target}_boot{i}.pkl")

            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_preds.append(y_pred)

        y_stack = np.vstack(y_preds)
        y_lower = np.min(y_stack, axis=0)
        y_upper = np.max(y_stack, axis=0)
        y_median = np.median(y_stack, axis=0)

        pd.DataFrame(
            {
                "Actual": y_test,
                "LowerBound": y_lower,
                "UpperBound": y_upper,
                "MedianPrediction": y_median,
            }
        ).to_excel(output_folder / f"Credal_Bounds_{target}.xlsx", index=False)

        if save_plots:
            _save_credal_plot(
                plt,
                y_true=y_test,
                y_lower=y_lower,
                y_upper=y_upper,
                y_median=y_median,
                y_all=y,
                target=target,
                model_label="SVR",
                plot_path=output_folder / f"Credal_Bounds_{target}.png",
            )

        summary.append(
            {
                "Target": target,
                "R2": r2_score(y_test, y_median),
                "RMSE": mean_squared_error(y_test, y_median, squared=False),
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_excel(
        output_folder.parent / "Credal_Model_Performance_SVR_Filtered_AllTuples.xlsx",
        index=False,
    )
    return summary_df


def train_gbr_credal_ensemble(
    df_features: pd.DataFrame,
    *,
    output_folder: Path = DEFAULT_GBR_FOLDER,
    n_credal: int = 30,
    test_size: float = 0.15,
    split_seed: int = 42,
    save_plots: bool = True,
) -> pd.DataFrame:
    deps = _import_ml_deps()
    joblib = deps["joblib"]
    plt = deps["plt"]
    train_test_split = deps["train_test_split"]
    GradientBoostingRegressor = deps["GradientBoostingRegressor"]
    resample = deps["resample"]
    r2_score = deps["r2_score"]
    mean_squared_error = deps["mean_squared_error"]

    output_folder.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, float | str]] = []

    for target in TARGETS:
        X = df_features[FEATURES].values
        y = df_features[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed
        )

        y_preds: list[np.ndarray] = []
        for i in range(n_credal):
            X_res, y_res = resample(X_train, y_train, replace=True, random_state=i)
            model = GradientBoostingRegressor(
                n_estimators=700,
                max_depth=3,
                learning_rate=0.015,
                subsample=0.85,
                random_state=i,
            )
            model.fit(X_res, y_res)
            joblib.dump(model, output_folder / f"credal_model_{target}_boot{i}.pkl")
            y_preds.append(model.predict(X_test))

        y_stack = np.vstack(y_preds)
        y_lower = np.min(y_stack, axis=0)
        y_upper = np.max(y_stack, axis=0)
        y_median = np.median(y_stack, axis=0)

        pd.DataFrame(
            {
                "Actual": y_test,
                "LowerBound": y_lower,
                "UpperBound": y_upper,
                "MedianPrediction": y_median,
            }
        ).to_excel(output_folder / f"Credal_Bounds_{target}.xlsx", index=False)

        if save_plots:
            _save_credal_plot(
                plt,
                y_true=y_test,
                y_lower=y_lower,
                y_upper=y_upper,
                y_median=y_median,
                y_all=y,
                target=target,
                model_label="GBR",
                plot_path=output_folder / f"Credal_Bounds_{target}.png",
            )

        summary.append(
            {
                "Target": target,
                "R2": r2_score(y_test, y_median),
                "RMSE": mean_squared_error(y_test, y_median, squared=False),
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_excel(
        output_folder.parent / "Credal_Model_Performance_GBR_Filtered_AllTuples.xlsx",
        index=False,
    )
    return summary_df


def run_training_workflow(
    *,
    excel_path: Path = DEFAULT_EXCEL_PATH,
    base_results_folder: Path = DEFAULT_BASE_RESULTS,
    model_types: Iterable[str] = ("svr", "gbr"),
    n_credal: int = 30,
    test_size: float = 0.15,
    split_seed: int = 42,
    apply_iqr_filter: bool = True,
    iqr_bin_width: float = 250.0,
    save_plots: bool = True,
) -> dict[str, pd.DataFrame]:
    tuple_results = base_results_folder / "Tuple_Data_Process"
    svr_folder = tuple_results / "ML_Surrogate_Credal_SVR"
    gbr_folder = tuple_results / "ML_Surrogate_Credal"
    tuple_results.mkdir(parents=True, exist_ok=True)

    df_features = prepare_training_dataframe(
        excel_path,
        apply_iqr_filter=apply_iqr_filter,
        bin_width=iqr_bin_width,
    )

    model_types_norm = {m.strip().lower() for m in model_types}
    out: dict[str, pd.DataFrame] = {}

    if "svr" in model_types_norm:
        out["svr"] = train_svr_credal_ensemble(
            df_features,
            output_folder=svr_folder,
            n_credal=n_credal,
            test_size=test_size,
            split_seed=split_seed,
            save_plots=save_plots,
        )
    if "gbr" in model_types_norm:
        out["gbr"] = train_gbr_credal_ensemble(
            df_features,
            output_folder=gbr_folder,
            n_credal=n_credal,
            test_size=test_size,
            split_seed=split_seed,
            save_plots=save_plots,
        )

    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train filtered-all-tuples surrogate credal ensembles (SVR/GBR)."
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=DEFAULT_EXCEL_PATH,
        help="Path to Yield_Results_by_Scenario.xlsx",
    )
    parser.add_argument(
        "--base-results-folder",
        type=Path,
        default=DEFAULT_BASE_RESULTS,
        help="Base output folder (Tuple_Data_Process is created under this).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["svr", "gbr"],
        choices=["svr", "gbr"],
        help="Model workflows to run.",
    )
    parser.add_argument(
        "--n-credal",
        type=int,
        default=30,
        help="Number of bootstrap models per target.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15, help="Test split ratio."
    )
    parser.add_argument(
        "--split-seed", type=int, default=42, help="Random seed for train/test split."
    )
    parser.add_argument(
        "--disable-iqr-filter",
        action="store_true",
        help="Disable all-tuple IQR filtering before training.",
    )
    parser.add_argument(
        "--iqr-bin-width",
        type=float,
        default=250.0,
        help="Bin width in mm for all-tuple IQR filtering.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation (Excel + model artifacts still saved).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    summaries = run_training_workflow(
        excel_path=args.excel_path,
        base_results_folder=args.base_results_folder,
        model_types=args.models,
        n_credal=args.n_credal,
        test_size=args.test_size,
        split_seed=args.split_seed,
        apply_iqr_filter=not args.disable_iqr_filter,
        iqr_bin_width=args.iqr_bin_width,
        save_plots=not args.no_plots,
    )

    for model_name, summary_df in summaries.items():
        print(f"✅ {model_name.upper()} completed: {len(summary_df)} targets")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
