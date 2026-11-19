"""
app.py
------
Interactive Streamlit dashboard for:
"Tumor-Fraction-Aware AI for Early Multi-Cancer Detection from Liquid Biopsy"

Features:
- Reads genuine benchmark outputs from results/metrics_by_tf.csv and results/predictions.csv.
- Renders:
  1. Sensitivity vs. Tumor Fraction analytical curves at 95% and 99% fixed specificity.
  2. Direct head-to-head performance comparisons against the ichorCNA baseline.
  3. Patient-level diagnostic distributions and cancer-type breakdown.
- Clear empty-state handler when evaluation runs have not yet been executed.
"""

import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="MCED cfDNA Liquid Biopsy Benchmark",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

RESULTS_DIR = Path("results")
METRICS_PATH = RESULTS_DIR / "metrics_by_tf.csv"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
CONFIG_PATH = Path("config/config.yaml")

# ------------------------------------------------------------------------------
# Sidebar Navigation & Context
# ------------------------------------------------------------------------------
st.sidebar.title("🧬 MCED cfDNA AI")
st.sidebar.markdown(
    """
    **Capstone Project**  
    *Tumor-Fraction-Aware AI for Early Multi-Cancer Detection from Liquid Biopsy*
    """
)
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **The Low-TF Sensitivity Gap:**  
    Standard sWGS methods (such as ichorCNA) collapse below 3% tumor fraction ($TF < 0.03$).
    This framework integrates multimodal fragmentomics (short-to-long fragment ratios, 4-mer motifs, and nucleosome protection)
    to rescue detection sensitivity down to 1% and 0.1% circulating tumor fraction.
    """
)

# ------------------------------------------------------------------------------
# Main Content
# ------------------------------------------------------------------------------
st.title("Tumor-Fraction-Aware AI for Early Multi-Cancer Detection")
st.markdown(
    "Evaluation benchmarks across in silico titrated cfDNA cohorts with zero-leakage patient-level cross-validation."
)

if not METRICS_PATH.exists() or os.path.getsize(METRICS_PATH) == 0:
    # --------------------------------------------------------------------------
    # Empty State: Pipeline has not been run yet
    # --------------------------------------------------------------------------
    st.warning("⚠️ No evaluation results found in `results/metrics_by_tf.csv`.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            ### Pipeline Evaluation Required
            Per strict scientific protocol, **no mock or hardcoded clinical results are displayed**.
            The dashboard visualizes genuine cross-validation outputs once the feature extraction and patient-grouped
            evaluation scripts finish execution.

            #### How to Generate Results:
            **Step 1:** Extract multimodal fragmentomic + ichorCNA features:
            ```bash
            python scripts/extract_features.py --mode merge --sample_sheet data/metadata/sample_sheet_template.csv --output_parquet data/processed/merged_features.parquet
            ```

            **Step 2:** Train zero-leakage patient-grouped models and benchmark against ichorCNA:
            ```bash
            python scripts/train_eval.py --features data/processed/merged_features.parquet --config config/config.yaml --output_dir results/
            ```

            **Step 3:** Refresh this browser tab to explore the generated benchmarks!
            """
        )

        st.markdown("---")
        st.subheader("Run Pilot Benchmark Now")
        st.markdown("Click below to execute the starter feature extraction and training pipeline directly:")

        if st.button("🚀 Run Pilot Feature Extraction & Patient-Grouped Benchmark", type="primary"):
            with st.spinner("Running extract_features.py and train_eval.py..."):
                try:
                    res1 = subprocess.run([
                        sys.executable, "scripts/extract_features.py",
                        "--mode", "merge",
                        "--sample_sheet", "data/metadata/sample_sheet_template.csv",
                        "--output_parquet", "data/processed/merged_features.parquet"
                    ], capture_output=True, text=True, check=True)

                    res2 = subprocess.run([
                        sys.executable, "scripts/train_eval.py",
                        "--features", "data/processed/merged_features.parquet",
                        "--config", "config/config.yaml",
                        "--output_dir", "results"
                    ], capture_output=True, text=True, check=True)

                    st.success("Pilot evaluation complete! Reloading results...")
                    st.rerun()
                except subprocess.CalledProcessError as err:
                    st.error(f"Pipeline execution failed: {err.stderr or err.stdout}")

    with col2:
        st.subheader("Target Tumor Fraction Tiers")
        tf_tiers_df = pd.DataFrame([
            {"Tier": "TF >= 10%", "Clinical Stage": "Advanced / Stage IV", "sWGS Feasibility": "High (Readily detectable)"},
            {"Tier": "5% <= TF < 10%", "Clinical Stage": "Stage III", "sWGS Feasibility": "Moderate (ichorCNA threshold)"},
            {"Tier": "1% <= TF < 5%", "Clinical Stage": "Early / Stage II", "sWGS Feasibility": "Poor without fragmentomics"},
            {"Tier": "0.1% <= TF < 1%", "Clinical Stage": "Very Early / Stage I", "sWGS Feasibility": "Fails standard CNA"}
        ])
        st.dataframe(tf_tiers_df, use_container_width=True, hide_index=True)

else:
    # --------------------------------------------------------------------------
    # Genuine Results Loaded
    # --------------------------------------------------------------------------
    metrics_df = pd.read_csv(METRICS_PATH)

    st.success(f"Loaded genuine evaluation benchmarks from `{METRICS_PATH}`.")

    # High-level KPIs
    col_a, col_b, col_c, col_d = st.columns(4)
    overall_rows = metrics_df[metrics_df["tumor_fraction_tier"] == "Overall Cohort"]

    best_model = "N/A"
    best_auc = 0.0
    if not overall_rows.empty:
        top_row = overall_rows.sort_values(by="auc_roc", ascending=False).iloc[0]
        best_model = top_row["model"]
        best_auc = top_row["auc_roc"]

    col_a.metric("Top Performing Model", best_model.split("(")[0].strip())
    col_b.metric("Overall AUC-ROC", f"{best_auc:.3f}")

    # Low-TF comparison at 1% <= TF < 5%
    low_tf_rows = metrics_df[metrics_df["tumor_fraction_tier"] == "1% <= TF < 5%"]
    if not low_tf_rows.empty:
        ichor_row = low_tf_rows[low_tf_rows["model"].str.contains("ichorCNA", case=False, na=False)]
        ai_row = low_tf_rows[~low_tf_rows["model"].str.contains("ichorCNA", case=False, na=False)].sort_values(
            by="sensitivity_at_95_spec", ascending=False
        )

        ichor_sens = ichor_row["sensitivity_at_95_spec"].values[0] if not ichor_row.empty else 0.0
        ai_sens = ai_row["sensitivity_at_95_spec"].values[0] if not ai_row.empty else 0.0

        col_c.metric("ichorCNA Sens @ 95% Spec (1-5% TF)", f"{ichor_sens * 100:.1f}%")
        col_d.metric("AI Sens @ 95% Spec (1-5% TF)", f"{ai_sens * 100:.1f}%", delta=f"{(ai_sens - ichor_sens) * 100:+.1f}%")

    st.markdown("---")

    # Tabbed views
    tab1, tab2, tab3 = st.tabs([
        "📈 Sensitivity vs. Tumor Fraction",
        "⚖️ Model vs. ichorCNA Baseline",
        "📋 Full Metric Tables & Cohort Predictions"
    ])

    with tab1:
        st.subheader("Sensitivity Across Circulating Tumor Fractions (Fixed 95% Specificity)")
        st.markdown(
            "This curve demonstrates how multimodal fragmentomic models rescue sensitivity in the critical "
            "low-fraction window ($0.1\% \le TF < 5\%$) where standard CNA calls drop to near zero."
        )

        tier_order = ["0.1% <= TF < 1%", "1% <= TF < 5%", "5% <= TF < 10%", "TF >= 10%"]
        filtered_tf_df = metrics_df[metrics_df["tumor_fraction_tier"].isin(tier_order)].copy()

        if not filtered_tf_df.empty:
            fig_tf = px.line(
                filtered_tf_df,
                x="tumor_fraction_tier",
                y="sensitivity_at_95_spec",
                color="model",
                markers=True,
                category_orders={"tumor_fraction_tier": tier_order},
                labels={
                    "tumor_fraction_tier": "Circulating Tumor Fraction Tier",
                    "sensitivity_at_95_spec": "Sensitivity at 95% Specificity",
                    "model": "Model Architecture"
                },
                title="Sensitivity vs. Tumor Fraction at Fixed 95% Specificity"
            )
            fig_tf.update_layout(yaxis=dict(range=[0, 1.05], tickformat=".0%"), template="plotly_white")
            st.plotly_chart(fig_tf, use_container_width=True)
        else:
            st.info("No tier-stratified results available in current metrics file.")

    with tab2:
        st.subheader("Head-to-Head Comparison: Multimodal AI vs. ichorCNA Baseline")
        st.markdown(
            "Comparison of sensitivity recovery at both **95% Specificity** (screening threshold) "
            "and **99% Specificity** (confirmatory clinical threshold)."
        )

        comp_cols = ["model", "tumor_fraction_tier", "sensitivity_at_95_spec", "sensitivity_at_99_spec", "auc_roc"]
        st.dataframe(
            metrics_df[comp_cols].style.format({
                "sensitivity_at_95_spec": "{:.1%}",
                "sensitivity_at_99_spec": "{:.1%}",
                "auc_roc": "{:.3f}"
            }),
            use_container_width=True
        )

    with tab3:
        st.subheader("Patient Predictions & Diagnostic Distribution")
        if PREDICTIONS_PATH.exists():
            preds_df = pd.read_csv(PREDICTIONS_PATH)
            st.markdown(f"Loaded out-of-fold predictions for {len(preds_df)} test samples across unique patients.")

            # Cancer type breakdown
            if "cancer_type" in preds_df.columns:
                fig_cancer = px.histogram(
                    preds_df,
                    x="cancer_type",
                    color="condition",
                    barmode="group",
                    title="Sample Distribution by Cancer Lineage"
                )
                st.plotly_chart(fig_cancer, use_container_width=True)

            st.dataframe(preds_df.head(25), use_container_width=True)
        else:
            st.info("Predictions file `results/predictions.csv` not yet generated.")
