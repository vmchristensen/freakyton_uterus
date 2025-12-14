import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
import os

# --- PORTABLE PATH SETUP ---
# Get the directory where the current script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Assume the project root is one level up (e.g., from 'data_cleaning' to 'freakyton_uterus')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 
# Define the data directory path
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# ---------------------------

np.random.seed(123)

# -------------------
# 1. LOAD DATA
# -------------------
file_path = os.path.join(DATA_DIR, "IQ_Cancer_Endometrio_cleaned.csv")
df = pd.read_csv(file_path)

# -------------------
# 2. DEFINE TIME & EVENTS
# -------------------
time_col = 'days_to_recurrence_or_death'

# RFS event
event_col_rfs = 'recurrence_death'

# OS event
df['death_event'] = (df['current_status'] == 2).astype(int)
event_col_os = 'death_event'

# -------------------
# 3. FEATURE ENGINEERING
# -------------------
def classify_histology(code):
    if pd.isna(code):
        return 0  # Assume Endometrioid
    if code == 2:
        return 0  # Endometrioid
    else:
        return 1  # Non-Endometrioid (high risk)

df['Non_Endometrioid'] = df['histologic_type'].apply(classify_histology)

# -------------------
# 4. SELECT PREOPERATIVE PREDICTORS
# -------------------
predictors = [
    'age',
    'BMI',
    'ASA_score',
    'grade',
    'myometrial_invasion',
    'distant_metastasis',
    'Non_Endometrioid'
]

df_model = df[predictors].copy()

# -------------------
# A. RECURRENCE-FREE SURVIVAL (RFS)
# -------------------
df_rfs = df_model.copy()
df_rfs[time_col] = df[time_col]
df_rfs[event_col_rfs] = df[event_col_rfs]

print("\n--- Fitting RFS Cox Model ---")
cph_rfs = CoxPHFitter(penalizer=0.1)
cph_rfs.fit(
    df_rfs,
    duration_col=time_col,
    event_col=event_col_rfs,
    show_progress=True
)

cph_rfs.print_summary()

# Risk scores & stratification
df['risk_score_rfs'] = cph_rfs.predict_partial_hazard(df_model)
median_risk_rfs = df['risk_score_rfs'].median()
df['risk_group_rfs'] = np.where(
    df['risk_score_rfs'] > median_risk_rfs,
    'High risk',
    'Low risk'
)

# Kaplan–Meier: RFS
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))

for group in ['Low risk', 'High risk']:
    mask = df['risk_group_rfs'] == group
    kmf.fit(
        durations=df.loc[mask, time_col],
        event_observed=df.loc[mask, event_col_rfs],
        label=group
    )
    kmf.plot_survival_function(ci_show=True)

plt.title("Recurrence-Free Survival (RFS) by Predicted Risk Group")
plt.xlabel("Days since diagnosis")
plt.ylabel("RFS Probability")
plt.grid(True)
plt.show()

# -------------------
# B. OVERALL SURVIVAL (OS)
# -------------------
df_os = df_model.copy()
df_os[time_col] = df[time_col]
df_os[event_col_os] = df[event_col_os]

print("\n--- Fitting OS Cox Model ---")
cph_os = CoxPHFitter(penalizer=0.1)
cph_os.fit(
    df_os,
    duration_col=time_col,
    event_col=event_col_os,
    show_progress=True
)

cph_os.print_summary()

# Risk scores & stratification
df['risk_score_os'] = cph_os.predict_partial_hazard(df_model)
median_risk_os = df['risk_score_os'].median()
df['risk_group_os'] = np.where(
    df['risk_score_os'] > median_risk_os,
    'High risk',
    'Low risk'
)

# Kaplan–Meier: OS
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))

for group in ['Low risk', 'High risk']:
    mask = df['risk_group_os'] == group
    kmf.fit(
        durations=df.loc[mask, time_col],
        event_observed=df.loc[mask, event_col_os],
        label=group
    )
    kmf.plot_survival_function(ci_show=True)

plt.title("Overall Survival (OS) by Predicted Risk Group")
plt.xlabel("Days since diagnosis")
plt.ylabel("OS Probability")
plt.grid(True)
plt.show()
