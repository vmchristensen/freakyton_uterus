import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PORTABLE PATH SETUP ---
# Get the directory where the current script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Assume the project root is one level up (e.g., from 'data_cleaning' to 'freakyton_uterus')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 
# Define the data directory path
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# ---------------------------

# --- CONFIGURATION ---
st.set_page_config(page_title="NSMP Risk Calculator", layout="wide")

# --- LOAD MODELS (Classification and Survival) ---
@st.cache_resource
def load_models():
    """Loads both the classification and survival models from file."""
    try:
        # FIX: Use SCRIPT_DIR to point to the models correctly
        clf_model_path = os.path.join(SCRIPT_DIR, 'nsmp_recurrence_model_preop.pkl')
        surv_model_path = os.path.join(SCRIPT_DIR, 'nsmp_survival_model_preop.pkl')
        
        clf_model = joblib.load(clf_model_path)
        surv_model = joblib.load(surv_model_path)
        return clf_model, surv_model
    except FileNotFoundError as e:
        # Note: Updated the error message to reflect the correct model names
        st.error(f"Model file not found: {e.filename}. Please ensure both 'nsmp_recurrence_model_preop.pkl' and 'nsmp_survival_model_preop.pkl' are in the same folder as this app.")
        st.stop()

clf_model, surv_model = load_models()

# --- UI DESIGN ---
st.title("🧬 NSMP Endometrial Cancer Risk Stratification")
st.markdown("""
**Target Population:** Patients with *Non-Specific Molecular Profile (NSMP)* Endometrial Cancer.
This tool integrates clinicopathological data to predict recurrence risk and recurrence-free survival probability at the time of diagnosis.""")
st.divider()

# --- INPUTS (Sidebar) ---
st.sidebar.header("Patient & Tumor Characteristics")

def get_user_input():
    """Gathers all necessary inputs from the user via the sidebar."""
    
    # --- 1. PATIENT FACTORS ---
    st.sidebar.subheader("Patient Factors")
    age = st.sidebar.slider("Age at Diagnosis", 20, 100, 65)
    bmi = st.sidebar.slider("BMI (kg/m²)", 15.0, 60.0, 28.0, 0.1)
    
    asa_dict = {"ASA I (Healthy)": 0, "ASA II (Mild Disease)": 1, "ASA III (Severe Disease)": 2, "ASA IV+": 3}
    asa_key = st.sidebar.select_slider("ASA Score", options=list(asa_dict.keys()))
    
    st.sidebar.markdown("---")

    # --- 2. TUMOR PATHOLOGY ---
    st.sidebar.subheader("Pathology")
    
    # This creates the 'Non_Endometrioid' feature
    histo_dict = {"Endometrioid": 0, "Non-Endometrioid (Serous, Clear Cell, etc.)": 1}
    histo_key = st.sidebar.selectbox("Definitive Histology Type", list(histo_dict.keys()))
    non_endo_val = histo_dict[histo_key]

    grade_dict = {"Low Grade (G1-G2)": 1, "High Grade (G3)": 2}
    grade_key = st.sidebar.selectbox("Grade", list(grade_dict.keys()))
    
    myo_dict = {"No Invasion": 0, "<50%": 1, "≥50%": 2, "Serosal Invasion": 3}
    myo_key = st.sidebar.selectbox("Myometrial Invasion", list(myo_dict.keys()))

    st.sidebar.markdown("---")
    
    # --- 3. SPREAD & BIOMARKERS ---
    st.sidebar.subheader("Spread & Biomarkers")

    # --- SELECTBOX ---
    binary_dict = {"Negative": 0, "Positive": 1}

    meta_key = st.sidebar.selectbox("Distant Metastasis at Diagnosis", list(binary_dict.keys()))
    meta_val = binary_dict[meta_key]

    # --- ASSEMBLE DATAFRAME ---
    # This list must match the order and names from your training script
    feature_names_in_order = [
    'age',
    'BMI',
    'ASA_score',
    'grade',
    'myometrial_invasion',
    'distant_metastasis',
    'Non_Endometrioid'
    ]

    
    data = {
        'age': age,
        'BMI': bmi,
        'ASA_score': asa_dict[asa_key],
        'grade': grade_dict[grade_key],
        'myometrial_invasion': myo_dict[myo_key],
        'distant_metastasis': meta_val,
        'Non_Endometrioid': non_endo_val
    }
    
    return pd.DataFrame(data, index=[0])[feature_names_in_order] # Enforce correct column order

input_df = get_user_input()

# --- PREDICTION LOGIC ---
if st.button("Calculate Risk Profile", type="primary"):
    
    # Prediction 1: Recurrence Probability (using Random Forest)
    prob = clf_model.predict_proba(input_df)[0][1]
    prob_pct = prob * 100

    # Prediction 2: Survival Curve (using Cox Model)
    survival_function = surv_model.predict_survival_function(input_df)
    survival_df = survival_function.reset_index().rename(columns={'index': 'Days', survival_function.columns[0]: 'Survival Probability'})
    survival_df['Months'] = survival_df['Days'] / 30.44

    # Risk Stratification Logic
    if prob < 0.20:
        risk_group, markdown_color, chart_color, rec, icon = "LOW RISK", "green","#2ecc71" ,"Observation Recommended", "✅"
    elif prob < 0.6:
        risk_group, markdown_color, chart_color,  rec, icon = "INTERMEDIATE RISK", "orange","#f39c12", "Consider Adjuvant Therapy (e.g., Brachytherapy)", "⚠️"
    else:
        risk_group, markdown_color, chart_color,  rec, icon = "HIGH RISK", "red", "#e74c3c", "Adjuvant Therapy Recommended (e.g., EBRT +/- Chemo)", "🚨"

    # --- DASHBOARD LAYOUT ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"### Recurrence Probability")
        st.metric(label="Estimated Risk of Recurrence", value=f"{prob_pct:.1f}%")
        st.progress(prob)
        st.markdown(f"### Classification: :{markdown_color}[{risk_group}] {icon}")
        st.info(f"**Recommendation:** {rec}")

    with col2:
        st.markdown("### Personalized Survival Projection")
        st.line_chart(survival_df, x='Months', y='Survival Probability', color=[chart_color])
        st.caption("Estimated probability of remaining recurrence-free over time for this patient profile.")

    # --- EXPLAINABILITY ---
    st.markdown("---")
    st.markdown("#### Key Risk Factors for this Patient")
    
    reasons = []
    if input_df['Non_Endometrioid'].iloc[0] == 1: reasons.append("⚠️ High-Risk Histology")
    if input_df['grade'].iloc[0] == 2: reasons.append("⚠️ High Grade (G3)")
    if input_df['myometrial_invasion'].iloc[0] >= 2: reasons.append("⚠️ Deep/Serosal Invasion")
    if input_df['distant_metastasis'].iloc[0] == 1: reasons.append("⚠️ Distant Metastasis Present")
    

    if not reasons:
        st.success("✅ No major high-risk features identified.")
    else:
        for r in reasons:
            st.write(r)

st.caption("Disclaimer: This tool is a prototype for research purposes only and not a substitute for clinical judgment.")