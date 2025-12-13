import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

# Load data
file_path = r"C:\Users\nadine jager\Documents\hackathon\freakyton_uterus\IQ_Cancer_Endometrio_cleaned.csv"
df = pd.read_csv(file_path)

print(df.isna().sum())


# Time and event columns
time_col = 'days_to_recurrence_or_death'
event_col = 'recurrence_death'  # 1 = death/recurrence, 0 = censored


# Pre-recurrence predictors (available at diagnosis)
predictors = [
    'age',
    'BMI',
    'histologic_type',
    'grade',
    'myometrial_invasion',
    'distant_metastasis',
    'risk_group_preSurgery',
    'genetic_study_1'
]

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df[predictors], drop_first=True)

# Add time and event columns
df_encoded[time_col] = df[time_col]
df_encoded[event_col] = df[event_col]

# Fit Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(df_encoded, duration_col=time_col, event_col=event_col)
cph.print_summary()  # optional: see coefficients and p-values

# Compute risk scores from the model
df['risk_score'] = cph.predict_partial_hazard(df_encoded)

# Split patients into low-risk and high-risk groups (median split)
median_risk = df['risk_score'].median()
df['risk_group'] = df['risk_score'].apply(lambda x: 'High risk' if x > median_risk else 'Low risk')

# Plot Kaplan-Meier curves
kmf = KaplanMeierFitter()
plt.figure(figsize=(10,6))

for group in ['Low risk', 'High risk']:
    mask = df['risk_group'] == group
    kmf.fit(durations=df[mask][time_col], event_observed=df[mask][event_col], label=group)
    kmf.plot_survival_function(ci_show=True)

plt.title("Kaplan-Meier Survival Curve by Risk Group at Diagnosis")
plt.xlabel("Days since diagnosis")
plt.ylabel("Survival probability")
plt.grid(True)
plt.show()
