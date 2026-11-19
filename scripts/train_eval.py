#!/usr/bin/env python3
"""
train_eval.py
-------------
Patient-grouped cross-validation training and sensitivity benchmarking engine
for early multi-cancer detection from cfDNA fragmentomic profiles.

Key Capabilities:
1. Zero-Leakage Patient Grouping: Uses GroupKFold keyed strictly on `patient_id`.
   Guarantees that all in silico dilutions, technical replicates, or aliquots
   from the same patient/donor remain strictly segregated in either train or test folds.
2. Multimodal Classifiers:
   - XGBoost
   - LightGBM
   - PyTorch Multi-Layer Perceptron (MLP)
3. Direct Baseline Comparison:
   - Benchmarks models directly against the ichorCNA baseline (standard CNA tumor fraction estimation).
4. Sensitivity at Fixed Specificity:
   - Computes empirical sensitivity at 95% and 99% specificity across target tumor fraction tiers:
     TF >= 10%, 5% <= TF < 10%, 1% <= TF < 5%, and 0.1% <= TF < 1%.
5. Real Clinical Metrics Output:
   - Writes evaluated metrics to results/metrics_by_tf.csv and results/predictions.csv.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ML Libraries with fallback handling
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("train_eval")


# ------------------------------------------------------------------------------
# PyTorch MLP Architecture
# ------------------------------------------------------------------------------
if HAS_TORCH:
    class FragMLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark tumor-fraction-aware MCED classifiers."
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/merged_features.parquet",
        help="Path to merged features parquet/csv."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to project config.yaml."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save metric tables and diagnostic plots."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to save trained model artifacts."
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logger.warning("Config file not found at %s. Using default settings.", config_path)
        return {
            "training": {"n_splits": 5, "fixed_specificities": [0.95, 0.99]},
            "in_silico_dilution": {"fractions": [0.10, 0.05, 0.01, 0.001]}
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_sensitivity_at_fixed_specificity(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_specificity: float = 0.95
) -> float:
    """Compute empirical sensitivity at or above target specificity.

    Specificity = TN / (TN + FP) = 1 - FPR
    Target specificity implies FPR <= (1 - target_specificity).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)
    # Find points where specificity >= target_specificity (FPR <= 1 - target_spec)
    max_allowable_fpr = 1.0 - target_specificity
    valid_indices = np.where(fpr <= max_allowable_fpr)[0]

    if len(valid_indices) == 0:
        return 0.0

    best_idx = valid_indices[-1]
    return float(tpr[best_idx])


def assign_tf_tier(tf: float) -> str:
    """Categorize continuous measured/diluted tumor fraction into analytical tiers."""
    if tf >= 0.10:
        return "TF >= 10%"
    elif tf >= 0.05:
        return "5% <= TF < 10%"
    elif tf >= 0.01:
        return "1% <= TF < 5%"
    elif tf >= 0.001:
        return "0.1% <= TF < 1%"
    else:
        return "TF < 0.1% / Baseline"


def train_mlp_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    epochs: int = 60,
    batch_size: int = 32
) -> np.ndarray:
    """Train PyTorch MLP on training fold and return validation probabilities."""
    if not HAS_TORCH:
        return np.zeros(len(X_val))

    torch.manual_seed(42)
    model = FragMLP(input_dim=input_dim)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(X_val, dtype=torch.float32)
        val_logits = model(val_x)
        val_probs = torch.sigmoid(val_logits).numpy()

    return val_probs


def run_patient_grouped_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5
) -> pd.DataFrame:
    """Run GroupKFold cross validation grouped strictly by patient_id."""
    # Check patient ID grouping
    groups = df["patient_id"].values
    n_unique_patients = len(np.unique(groups))
    logger.info("Enforcing zero-leakage GroupKFold across %d unique patients.", n_unique_patients)

    actual_splits = min(n_splits, n_unique_patients)
    if actual_splits < 2:
        raise ValueError(f"Need at least 2 unique patient groups for CV; found {n_unique_patients}")

    gkf = GroupKFold(n_splits=actual_splits)

    X = df[feature_cols].values
    y = df["is_cancer"].values

    # Pre-allocate prediction columns
    preds_df = df.copy()
    preds_df["pred_xgboost"] = np.nan
    preds_df["pred_lightgbm"] = np.nan
    preds_df["pred_mlp"] = np.nan
    preds_df["pred_ichorcna"] = df.get("ichor_tf", np.zeros(len(df))).values

    scaler = StandardScaler()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        # Verification: check for zero overlap in patient IDs
        train_patients = set(df.iloc[train_idx]["patient_id"])
        val_patients = set(df.iloc[val_idx]["patient_id"])
        overlap = train_patients.intersection(val_patients)
        assert len(overlap) == 0, f"DATA LEAKAGE DETECTED in fold {fold}: {overlap}"

        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_val_raw, y_val = X[val_idx], y[val_idx]

        # Fit scaler ONLY on train fold, preserving DataFrame structure
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_val_df = pd.DataFrame(X_val_scaled, columns=feature_cols)

        # 1. XGBoost
        if HAS_XGB:
            clf_xgb = xgb.XGBClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, eval_metric="logloss"
            )
            clf_xgb.fit(X_train_df, y_train)
            preds_df.iloc[val_idx, preds_df.columns.get_loc("pred_xgboost")] = clf_xgb.predict_proba(X_val_df)[:, 1]

        # 2. LightGBM
        if HAS_LGB:
            clf_lgb = lgb.LGBMClassifier(
                n_estimators=100, max_depth=2, num_leaves=7, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                min_child_samples=1, random_state=42, verbose=-1
            )
            clf_lgb.fit(X_train_df, y_train)
            preds_df.iloc[val_idx, preds_df.columns.get_loc("pred_lightgbm")] = clf_lgb.predict_proba(X_val_df)[:, 1]

        # 3. PyTorch MLP
        if HAS_TORCH:
            mlp_probs = train_mlp_fold(X_train_scaled, y_train, X_val_scaled, input_dim=X.shape[1])
            preds_df.iloc[val_idx, preds_df.columns.get_loc("pred_mlp")] = mlp_probs

    return preds_df


def compute_metrics_table(
    preds_df: pd.DataFrame,
    model_col: str,
    model_name: str
) -> List[Dict]:
    """Compute sensitivity at 95% and 99% specificity by tumor fraction tier."""
    results = []

    valid_mask = ~preds_df[model_col].isna()
    sub_df = preds_df[valid_mask].copy()

    if len(sub_df) == 0:
        return results

    y_true = sub_df["is_cancer"].values
    y_scores = sub_df[model_col].values

    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        return results

    overall_auc = roc_auc_score(y_true, y_scores)
    sub_df["tf_tier"] = sub_df["measured_tf"].apply(assign_tf_tier)

    # Calculate global threshold for 95% and 99% specificity
    healthy_mask = (y_true == 0)
    if np.sum(healthy_mask) > 0:
        healthy_scores = y_scores[healthy_mask]
        thresh_95 = float(np.percentile(healthy_scores, 95))
        thresh_99 = float(np.percentile(healthy_scores, 99))
    else:
        thresh_95 = 0.5
        thresh_99 = 0.5

    # Overall cohort metric
    cancer_mask = (y_true == 1)
    sens_95_all = float(np.mean(y_scores[cancer_mask] >= thresh_95)) if np.sum(cancer_mask) > 0 else 0.0
    sens_99_all = float(np.mean(y_scores[cancer_mask] >= thresh_99)) if np.sum(cancer_mask) > 0 else 0.0

    results.append({
        "model": model_name,
        "tumor_fraction_tier": "Overall Cohort",
        "sensitivity_at_95_spec": round(sens_95_all, 4),
        "sensitivity_at_99_spec": round(sens_99_all, 4),
        "auc_roc": round(overall_auc, 4),
        "sample_count": len(sub_df)
    })

    # Breakdown by tumor fraction tiers
    for tier in ["TF >= 10%", "5% <= TF < 10%", "1% <= TF < 5%", "0.1% <= TF < 1%"]:
        tier_cancer = sub_df[(sub_df["tf_tier"] == tier) & (sub_df["is_cancer"] == 1)]
        if len(tier_cancer) > 0:
            sens_95 = float(np.mean(tier_cancer[model_col].values >= thresh_95))
            sens_99 = float(np.mean(tier_cancer[model_col].values >= thresh_99))
            results.append({
                "model": model_name,
                "tumor_fraction_tier": tier,
                "sensitivity_at_95_spec": round(sens_95, 4),
                "sensitivity_at_99_spec": round(sens_99, 4),
                "auc_roc": round(overall_auc, 4),
                "sample_count": len(tier_cancer)
            })

    return results


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    n_splits = config.get("training", {}).get("n_splits", 5)

    if not os.path.exists(args.features):
        logger.error("Features matrix not found: %s", args.features)
        logger.info("Run 'python scripts/extract_features.py --mode merge' first.")
        sys.exit(1)

    if args.features.endswith(".parquet"):
        df = pd.read_parquet(args.features)
    else:
        df = pd.read_csv(args.features)

    logger.info("Loaded feature dataset: %d samples, %d total columns", len(df), df.shape[1])

    # Identify numerical feature columns
    metadata_cols = {
        "sample_id", "patient_id", "condition", "cancer_type", "stage",
        "measured_tf", "is_cancer", "pred_xgboost", "pred_lightgbm", "pred_mlp", "pred_ichorcna"
    }
    feature_cols = [c for c in df.columns if c not in metadata_cols and np.issubdtype(df[c].dtype, np.number)]
    logger.info("Using %d fragmentomic and CNA feature columns for training.", len(feature_cols))

    # Run zero-leakage Patient-Grouped Cross-Validation
    preds_df = run_patient_grouped_cv(df, feature_cols, n_splits=n_splits)

    # Save out-of-fold predictions
    preds_out_path = os.path.join(args.output_dir, "predictions.csv")
    preds_df.to_csv(preds_out_path, index=False)
    logger.info("Saved out-of-fold predictions to %s", preds_out_path)

    # Compute metrics for all models vs ichorCNA baseline
    all_metrics = []
    models_to_eval = [
        ("pred_ichorcna", "ichorCNA Baseline"),
        ("pred_xgboost", "XGBoost (Fragmentomics + CNA)"),
        ("pred_lightgbm", "LightGBM (Fragmentomics + CNA)"),
        ("pred_mlp", "PyTorch MLP (Fragmentomics + CNA)"),
    ]

    for col, name in models_to_eval:
        if col in preds_df.columns and not preds_df[col].isna().all():
            m = compute_metrics_table(preds_df, col, name)
            all_metrics.extend(m)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = os.path.join(args.output_dir, "metrics_by_tf.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info("Successfully generated metrics summary: %s", metrics_csv_path)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS BY TUMOR FRACTION TIER (Fixed Specificity = 95% & 99%)")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
