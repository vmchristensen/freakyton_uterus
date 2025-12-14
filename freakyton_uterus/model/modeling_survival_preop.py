import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
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

np.random.seed(123)
# 1. LOAD & PREP DATA
# -------------------
file_path = os.path.join(DATA_DIR, "IQ_Cancer_Endometrio_cleaned.csv")
df = pd.read_csv(file_path)

# --- A. Prepare RFS Columns ---
df['RFS_Event'] = df['recurrence']
# Convert days to months for better readability
df['RFS_Time'] = df['days_to_recurrence_or_death'] / 30.44 

# --- B. Prepare OS Columns ---
# Map current_status (1=Alive, 2=Deceased) to Binary (0=Alive, 1=Dead)
df['OS_Event'] = df['current_status'].apply(lambda x: 1 if x == 2 else 0)

# Calculate OS Time (Needs Date handling)
# Ensure columns are datetime
date_cols = ['diagnosis_date', 'death_date', 'last_visit_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Logic: If dead, use death date. If alive, use last visit.
def calc_os_time(row):
    start_date = row['diagnosis_date']
    if row['OS_Event'] == 1: # Dead
        end_date = row['death_date']
    else: # Alive
        end_date = row['last_visit_date']
    
    if pd.isnull(end_date) or pd.isnull(start_date):
        return row['days_to_recurrence_or_death'] / 30.44 # Fallback
        
    return (end_date - start_date).days / 30.44

df['OS_Time'] = df.apply(calc_os_time, axis=1)

# Drop any negative or missing times (bad data)
df = df[df['RFS_Time'] > 0]
df = df[df['OS_Time'] > 0]

def classify_histology(code):
    if pd.isna(code): 
        return 0 # Assume Endometrioid (Most common) if missing
    if code == 2: 
        return 0 # Endometrioid (Standard)
    else: 
        return 1 # Non-Endometrioid (Serous, Clear Cell, etc. -> High Risk)


# 2. KAPLAN-MEIER CURVES (The Visuals)
# ------------------------------------
kmf = KaplanMeierFitter()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot RFS
kmf.fit(df['RFS_Time'], df['RFS_Event'], label='All Patients')
kmf.plot(ax=axes[0], show_censors=True, color="#2ecc71")
axes[0].set_title("Recurrence-Free Survival (RFS)")
axes[0].set_xlabel("Months since Diagnosis")
axes[0].set_ylabel("Survival Probability")
axes[0].set_ylim(0, 1.05)

# Plot OS
kmf.fit(df['OS_Time'], df['OS_Event'], label='All Patients')
kmf.plot(ax=axes[1], show_censors=True, color="#3498db")
axes[1].set_title("Overall Survival (OS)")
axes[1].set_xlabel("Months since Diagnosis")
axes[1].set_ylabel("Survival Probability")
axes[1].set_ylim(0, 1.05)

plt.show()


# 3. TRAIN COX MODEL (For Predictions)
# ------------------------------------
# Select the same features as your Random Forest
# Create the new feature
df['Non_Endometrioid'] = df['histologic_type'].apply(classify_histology)

# --- UPDATED FEATURE LIST ---
features = [
    'age',
    'BMI',
    'ASA_score',
    'grade',
    'myometrial_invasion',
    'distant_metastasis',
    'Non_Endometrioid'  # <--- ADDED BACK
]


X_surv = df[features].copy()
X_surv['RFS_Time'] = df['RFS_Time']
X_surv['RFS_Event'] = df['RFS_Event']

# Fit the Model
cph = CoxPHFitter(penalizer=0.1) # Penalizer helps with small data/collinearity
cph.fit(X_surv, duration_col='RFS_Time', event_col='RFS_Event')

# Print the Hazard Ratios (Risk Factors)
print("--- HAZARD RATIOS (Risk Multipliers) ---")
print(cph.summary[['exp(coef)', 'p']])

joblib.dump(cph, os.path.join(SCRIPT_DIR, 'nsmp_survival_model_preop.pkl'))
print("SUCCESS! Cox Survival model saved as 'nsmp_survival_model.pkl'")

