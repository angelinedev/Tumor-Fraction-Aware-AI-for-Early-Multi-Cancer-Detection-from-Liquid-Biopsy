"""
test_dilution.py
----------------
Unit and integration tests for:
1. In silico titration mathematical mixing ratios and edge cases.
2. Zero-leakage patient-level GroupKFold isolation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, KFold

# Add repository root to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.in_silico_dilution import calculate_mixing_counts


class TestInSilicoDilutionCalculations:
    """Tests for dilution arithmetic, fractions, and boundary conditions."""

    def test_standard_dilution_ratios(self):
        total_reads = 20_000_000

        # 10% target TF from 100% cancer baseline
        n_cancer, n_healthy, alpha = calculate_mixing_counts(0.10, 1.0, total_reads)
        assert n_cancer == 2_000_000
        assert n_healthy == 18_000_000
        assert n_cancer + n_healthy == total_reads
        assert pytest.approx(alpha, rel=1e-5) == 0.10

        # 5% target TF
        n_cancer, n_healthy, alpha = calculate_mixing_counts(0.05, 1.0, total_reads)
        assert n_cancer == 1_000_000
        assert n_healthy == 19_000_000
        assert n_cancer + n_healthy == total_reads

        # 1% target TF (Critical low-TF boundary)
        n_cancer, n_healthy, alpha = calculate_mixing_counts(0.01, 1.0, total_reads)
        assert n_cancer == 200_000
        assert n_healthy == 19_800_000
        assert n_cancer + n_healthy == total_reads

        # 0.1% target TF (Extreme low-fraction sensitivity limit)
        n_cancer, n_healthy, alpha = calculate_mixing_counts(0.001, 1.0, total_reads)
        assert n_cancer == 20_000
        assert n_healthy == 19_980_000
        assert n_cancer + n_healthy == total_reads

    def test_non_pure_cancer_baseline(self):
        """When cancer donor cfDNA has 20% baseline TF, a 1% target requires alpha = 0.01 / 0.20 = 0.05."""
        total_reads = 1_000_000
        n_cancer, n_healthy, alpha = calculate_mixing_counts(
            target_tf=0.01,
            cancer_baseline_tf=0.20,
            total_reads=total_reads
        )
        assert pytest.approx(alpha, rel=1e-5) == 0.05
        assert n_cancer == 50_000
        assert n_healthy == 950_000

    def test_edge_cases(self):
        total_reads = 1_000_000
        # Zero tumor fraction (pure healthy)
        n_c, n_h, alpha = calculate_mixing_counts(0.0, 1.0, total_reads)
        assert n_c == 0
        assert n_h == total_reads

        # Target TF equal to or exceeding baseline (capped to 100% cancer)
        n_c, n_h, alpha = calculate_mixing_counts(0.5, 0.5, total_reads)
        assert n_c == total_reads
        assert n_h == 0

        # Invalid cancer baseline
        with pytest.raises(ValueError):
            calculate_mixing_counts(0.01, 0.0, total_reads)


class TestPatientLeakagePrevention:
    """Verifies that patient-level grouping strictly prevents read and replicate leakage."""

    @pytest.fixture
    def multi_replicate_patient_dataset(self) -> pd.DataFrame:
        """Create a synthetic cohort with multiple diluted replicates per patient."""
        records = []
        # 10 cancer patients, each with 4 dilution replicates (10%, 5%, 1%, 0.1%)
        for p_idx in range(1, 11):
            p_id = f"PATIENT_CANCER_{p_idx:03d}"
            for tf in [0.10, 0.05, 0.01, 0.001]:
                records.append({
                    "sample_id": f"{p_id}_TF_{tf}",
                    "patient_id": p_id,
                    "condition": "Cancer",
                    "measured_tf": tf,
                    "is_cancer": 1,
                    "feat_1": np.random.randn(),
                    "feat_2": np.random.randn(),
                })

        # 10 healthy control donors (1 sample each)
        for h_idx in range(1, 11):
            p_id = f"PATIENT_HEALTHY_{h_idx:03d}"
            records.append({
                "sample_id": f"{p_id}_REP1",
                "patient_id": p_id,
                "condition": "Healthy",
                "measured_tf": 0.0,
                "is_cancer": 0,
                "feat_1": np.random.randn(),
                "feat_2": np.random.randn(),
            })

        return pd.DataFrame(records)

    def test_group_kfold_guarantees_zero_patient_overlap(self, multi_replicate_patient_dataset):
        df = multi_replicate_patient_dataset
        groups = df["patient_id"].values
        gkf = GroupKFold(n_splits=5)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), 1):
            train_patients = set(df.iloc[train_idx]["patient_id"])
            val_patients = set(df.iloc[val_idx]["patient_id"])

            overlap = train_patients.intersection(val_patients)
            assert len(overlap) == 0, f"Fold {fold} leaked patients: {overlap}"

    def test_standard_kfold_causes_severe_leakage(self, multi_replicate_patient_dataset):
        """Demonstrates that standard naive K-Fold leaks patient replicates into validation."""
        df = multi_replicate_patient_dataset
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        leakage_detected = False
        for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
            train_patients = set(df.iloc[train_idx]["patient_id"])
            val_patients = set(df.iloc[val_idx]["patient_id"])
            overlap = train_patients.intersection(val_patients)
            if len(overlap) > 0:
                leakage_detected = True
                break

        assert leakage_detected, "Naive KFold should cause patient leakage across dilution replicates!"
