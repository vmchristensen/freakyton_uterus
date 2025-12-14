# 🏥 Freakyton Uterus: NSMP Endometrial Cancer Prognostic Model

This repository hosts a complete Machine Learning pipeline for predicting the **Recurrence Risk** and **Recurrence-Free Survival (RFS)** for patients with the Non-Specific Molecular Profile (NSMP) subtype of Endometrial Cancer.

The entire workflow, from raw data ingestion to model training and deployment assets, is automated and designed to be portable across different local environments.

---

## 🚀 Getting Started

### 1. Project Structure

The project uses the following organized structure, with `pipeline.py` at the root acting as the main execution script.

| Directory | Purpose | Key Files |
| :--- | :--- | :--- |
| **`data/`** | Stores raw and intermediate data files (e.g., `IQ_Cancer_Endometrio_merged_NMSP.xlsx`). |
| **`data_cleaning/`** | Scripts for data ingestion, translation, and final cleaning. | `read_data.py`, `df_selection_translation.py`, `data_cleaning.py` |
| **`model/`** | Contains modeling scripts, the Streamlit app, and trained `.pkl` assets. | `modeling_binaryrecurrence_preop.py`, `modeling_survival_preop.py`, `app_preop.py` |
| **`eda/`** | Exploratory Data Analysis and feature selection. |
| **`pipeline.py`** | **The main script** that executes the entire workflow in sequence. |

### 2. Dependencies

Ensure you have Python 3.8+ and install all required libraries (e.g., `pandas`, `sklearn`, `joblib`, `lifelines`, `shap`, `streamlit`). A `requirements.txt` file should be generated for this step.

Bash
pip install -r requirements.txt

## ⚙️ Running the Pipeline
The entire process is streamlined into a single command, ensuring all data loading, cleaning, and model saving is handled using portable, relative file paths.

### 1. Execute the full workflow (Data to Model)
From the root directory of the repository, execute the main pipeline script:

Bash
python pipeline.py

This command sequentially runs the following core steps:
- Data Ingestion (read_data.py)
- Translation & Selection (df_selection_translation.py)
- Data Cleaning & Feature Engineering (data_cleaning.py)
- Binary Model Training (modeling_binaryrecurrence_preop.py) → Saves nsmp_recurrence_model_preop.pkl
- Survival Model Training (modeling_survival_preop.py) → Saves nsmp_survival_model_preop.pkl
- Plot Generation (modeling_survival_preop_plots.py)

### 2. Launch the Web Application
Once the models are successfully saved in the model/ directory, you can launch the interactive risk calculator:

Bash
streamlit run model/app_preop.py

## 📊 Methodology Highlights
- Objective: Risk stratification for NSMP Endometrial Cancer patients pre-operatively or immediately post-operatively.
- Classification Model: Random Forest Classifier (predicts recurrence probability).
- Survival Model: Cox Proportional Hazards Model (predicts time-to-RFS).

All file paths within the Python scripts are configured using the os module (os.path.join(DATA_DIR, ...) and os.path.join(SCRIPT_DIR, ...)), making the project independent of the user's local directory structure.
