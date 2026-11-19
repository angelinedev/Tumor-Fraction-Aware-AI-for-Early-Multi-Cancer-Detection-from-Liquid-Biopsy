# Tumor-Fraction-Aware AI for Early Multi-Cancer Detection from Liquid Biopsy

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 5/5 Passing](https://img.shields.io/badge/Tests-5%2F5%20Passing-brightgreen.svg)](tests/)

An open, reproducible machine learning pipeline designed to **rescue cancer detection sensitivity when tumor DNA in the blood is extremely low ($<1\%$)**.

---

## 📌 The Problem in Simple Words

- **Liquid Biopsy:** When cells die, they release tiny cut-up pieces of DNA into our bloodstream (cell-free DNA or **cfDNA**). In cancer patients, tumors also shed DNA fragments into the blood. Testing blood is cheap, fast, and non-invasive.
- **The Challenge (The Low Tumor-Fraction Gap):**
  - In **late-stage cancer** (Stages III–IV), $5\%\text{--}20\%$ of the DNA in the blood comes from the tumor. Traditional tools easily spot this.
  - In **early-stage cancer** (Stages I–II), **less than $1\%$ (often $0.1\%$)** of the DNA in the blood comes from the tumor. The rest ($99.9\%$) is healthy blood DNA!
  - **Traditional clinical tools (like `ichorCNA`) fail completely** at these low levels, reporting that the patient is healthy (False Negative).

```
   Tumor Fraction in Blood:
   [======================= 10% =======================]  --> Late Stage (All tools detect easily)
   [============ 5% ============]                          --> Mid Stage (Traditional tools struggle)
   [==== 1% ====]                                          --> Early Stage (Traditional tools FAIL < 20%)
   [= 0.1% =]                                              --> Very Early Stage (Traditional tools = 0%)
                                                                ▲
                                            🎯 THIS PROJECT'S TARGET
```

---

## 💡 Our Solution

Instead of looking only at large copy-number changes, we combine **multiple biological signals (Multimodal AI)**:
1. **DNA Fragment Lengths (DELFI):** Tumor DNA fragments are shorter ($100\text{--}150\text{ bp}$) than healthy blood DNA ($167\text{ bp}$).
2. **DNA End Motifs:** The 4-letter alphabet at the cut ends of DNA fragments (e.g., `CCCA`, `CCAA`) shifts due to cancer-specific nuclease enzymes.
3. **Copy Number Variation (ichorCNA):** Chromosomal gains and losses.
4. **Machine Learning Ensemble:** Combining **XGBoost**, **LightGBM**, and a **PyTorch Deep Neural Network** to catch weak signals without false alarms.
5. **Zero Data Leakage:** Grouping strictly by `patient_id` so blood tests from the same person never appear in both training and testing.

---

## 📁 Repository Structure

```
mced-cfdna/
├── config/
│   └── config.yaml                     # Central project settings & thresholds
├── dashboard/
│   └── app.py                          # Streamlit interactive live demo
├── data/
│   ├── metadata/
│   │   ├── SraRunTable.csv             # Raw SRA metadata from NCBI
│   │   └── sample_sheet.csv            # Clean manifest: 69 blood samples (62 unique patients)
│   └── processed/
│       └── merged_features.parquet     # Extracted multimodal feature table (69 samples x 272 features)
├── notebooks/
│   ├── eda/                            # Exploratory Data Analysis (Plain English)
│   │   ├── 01_cfdna_prjna795275_eda.ipynb          # Blood cfDNA dataset breakdown
│   │   ├── 02_tcga_cnv_reference_eda.ipynb         # Solid tumor tissue reference
│   │   └── 03_cross_dataset_comparison_eda.ipynb   # Blood vs. Tissue comparison
│   └── models/                         # Step-by-Step Model Building & Ensembling
│       ├── 01_model_xgboost.ipynb                  # Model 1: XGBoost Classifier
│       ├── 02_model_lightgbm.ipynb                 # Model 2: LightGBM Classifier
│       ├── 03_model_pytorch_mlp.ipynb              # Model 3: PyTorch Deep Neural Net
│       └── 04_model_ensembling_and_selection.ipynb # Model 4: Ensembling & Best Model Selection
├── references/
│   └── tcga_cnv/                       # 500 real TCGA solid tumor segment profiles
├── results/
│   ├── eda/                            # Diagnostic figures
│   ├── metrics_by_tf.csv               # Sensitivity at 95% & 99% specificity
│   └── predictions.csv                 # Out-of-fold patient predictions
├── scripts/
│   ├── download_sra.py                 # SRA sequencing downloader
│   ├── download_tcga.py                # GDC TCGA CNV downloader
│   ├── exploratory_analysis.py         # Automated EDA script
│   ├── extract_features.py             # Feature extractor (ichorCNA + FinaleToolkit)
│   ├── in_silico_dilution.py           # In silico titration engine (10%, 5%, 1%, 0.1% TF)
│   └── train_eval.py                   # Zero-leakage GroupKFold training engine
├── tests/
│   └── test_dilution.py                # Unit test suite (5/5 passing)
├── Dockerfile                          # Reproducible Linux container
├── environment.yml                     # Conda environment
├── requirements.txt                    # Pip dependencies
└── README.md
```

---

## 📊 Datasets Used

| Dataset | Type | Samples | What it is used for |
| :--- | :--- | :--- | :--- |
| **PRJNA795275** | Blood Plasma cfDNA | **69 runs (62 patients)** | Real liquid biopsy samples: 28 healthy controls, 29 early screen-detected lung nodules (`LDCT`), and 11 small-cell lung cancer (`SCLC`) cases. |
| **TCGA (GDC)** | Solid Tumor Tissue | **500 tumor profiles** | Pure tumor tissue reference (296 Lung `TCGA-LUAD`, 156 Kidney `TCGA-KIRC`, 48 Brain `TCGA-GBM`) to define true chromosomal alteration boundaries. |

---

## 👥 Quick Team Setup Guide (How Any Teammate Can Run in 3 Minutes)

If you are a team member working on this project, follow these 3 simple steps to get everything running on your computer:

### Step 1: Clone the GitHub Repository
```bash
git clone https://github.com/angelinedev/Tumor-Fraction-Aware-AI-for-Early-Multi-Cancer-Detection-from-Liquid-Biopsy.git
cd Tumor-Fraction-Aware-AI-for-Early-Multi-Cancer-Detection-from-Liquid-Biopsy
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Extract the Team Data Bundle (`mced_project_data_bundle.zip`)
To avoid downloading large genomic databases individually, download the **3.0 MB** `mced_project_data_bundle.zip` (shared by the team lead) and extract it directly into the repository root folder.

It automatically unpacks:
- `data/metadata/sample_sheet.csv` (all 69 patient runs)
- `data/processed/merged_features.parquet` (pre-computed 272 features)
- `references/tcga_cnv/` (all 500 TCGA lung/kidney/brain tumor segment files)
- `results/` (pre-computed metrics, figures, and predictions)

*(Note: To re-generate this zip file at any time from source, simply run `python scripts/package_team_data.py`).*

### Step 4: Open and Run Any Notebook or App!
Now you can immediately run:
- **Jupyter Notebooks:** Open any notebook in [`notebooks/eda/`](notebooks/eda/) or [`notebooks/models/`](notebooks/models/).
- **Live Streamlit Dashboard:** Run `streamlit run dashboard/app.py` in your terminal.
- **Unit Tests:** Run `pytest tests/ -v` (all 5 tests should pass).

---

## 🚀 Pipeline CLI Execution

### 1. Set Up Your Environment
```bash
# Clone the repository
git clone https://github.com/angelinedev/Tumor-Fraction-Aware-AI-for-Early-Multi-Cancer-Detection-from-Liquid-Biopsy.git
cd Tumor-Fraction-Aware-AI-for-Early-Multi-Cancer-Detection-from-Liquid-Biopsy

# Install dependencies
pip install -r requirements.txt

# Run the unit tests to verify your setup
pytest tests/ -v
```

### 2. Run the Feature Extraction & Zero-Leakage Training
```bash
# Extract multimodal features from the sample sheet
python scripts/extract_features.py --mode merge --sample_sheet data/metadata/sample_sheet.csv --output_parquet data/processed/merged_features.parquet

# Train models with patient-grouped cross-validation
python scripts/train_eval.py --features data/processed/merged_features.parquet --config config/config.yaml --output_dir results/
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🧪 Model Progression & Ensembling

All models are trained using **5-fold `GroupKFold` on `patient_id`** to guarantee that technical replicates or serial blood samples from the same person never leak into the validation test fold.

1. **`ichorCNA` (Clinical Baseline):** Standard copy-number thresholding. High specificity, but sensitivity drops to $0\%$ when tumor fraction falls below $3\%$.
2. **`XGBoost`:** Gradient-boosted decision trees discovering non-linear cutoffs between fragment length ratios and end motifs.
3. **`LightGBM`:** Fast leaf-wise boosting capturing multi-dimensional motif frequency interactions.
4. **`PyTorch Deep MLP`:** 3-layer neural network learning smooth continuous probability curves.
5. **`Weighted Ensemble` ($40\%$ XGB + $40\%$ LGB + $20\%$ MLP):** Combines tree-based rules with neural network representations, achieving the highest overall ROC-AUC and rescuing sensitivity at early stages ($0.1\%\text{--}1.0\%$ tumor fraction).

Open [`notebooks/models/04_model_ensembling_and_selection.ipynb`](notebooks/models/04_model_ensembling_and_selection.ipynb) to view the complete comparison tables and curves.

---

## 📚 Key References

1. **Science Bulletin (2025):** *Liquid biopsy-based multi-cancer early detection: an exploration road from evidence to implementation.* DOI: 10.1016/j.scib.2025.06.030.
2. **Cristiano, S., et al. (Nature 2019):** *Genome-wide cell-free DNA fragmentation in patients with cancer.* (DELFI method).
3. **Adalsteinsson, V. A., et al. (Nature Communications 2017):** *Scalable whole-exome sequencing of cell-free DNA reveals high concordance with metastatic tumors.* (ichorCNA method).
4. **FinaleToolkit (Bioinformatics Advances 2025):** High-speed computational toolkit for cfDNA fragmentomics.
