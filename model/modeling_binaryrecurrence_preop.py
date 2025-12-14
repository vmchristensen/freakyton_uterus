import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import shap
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

# Load data
file_path = os.path.join(DATA_DIR, "IQ_Cancer_Endometrio_cleaned.csv")
df = pd.read_csv(file_path)

y = df['recurrence_death']  # 0 or 1
y_time = df['days_to_recurrence_or_death'] # Time

def classify_histology(code):
    if pd.isna(code): 
        return 0 # Assume Endometrioid (Most common) if missing
    if code == 2: 
        return 0 # Endometrioid (Standard)
    else: 
        return 1 # Non-Endometrioid (Serous, Clear Cell, etc. -> High Risk)

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


# Create the clean X dataframe
X = df[features].copy()
# 2. TRAIN/TEST SPLIT
# -------------------
# Stratify ensures the % of recurrences is the same in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# 3. TRAIN THE MODEL
# ------------------
rf_model = RandomForestClassifier(
    n_estimators=200,       # Number of trees
    max_depth=5,            # Keep trees shallow to prevent overfitting
    class_weight='balanced',# Crucial for imbalanced medical data!
    random_state=42
)

rf_model.fit(X_train, y_train)

# 4. PREDICT PROBABILITIES (Not just Yes/No)
# ------------------------------------------
# We want the probability of Class 1 (Recurrence)
y_probs = rf_model.predict_proba(X_test)[:, 1]
y_pred = rf_model.predict(X_test)



# 5. GENERATE METRICS
# -------------------
auc_score = roc_auc_score(y_test, y_probs)

print(f"--- MODEL PERFORMANCE ---")
print(f"AUC Score: {auc_score:.3f}") # > 0.7 is good, > 0.8 is great
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. PLOT ROC CURVE
# -----------------
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Model Performance: Detecting Recurrence in NSMP')
plt.legend()
plt.show()


# ... (Previous model training code remains the same) ...

# 8. SHAP EXPLAINABILITY (The Fix)
# --------------------------------

# Initialize JS
shap.initjs()

# Create explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# ROBUST SHAP PLOTTING LOGIC
# Check if shap_values is a list or a 3D array to handle the format correctly
if isinstance(shap_values, list):
    # Old format: List of [Class0, Class1] -> We want Class 1 (Recurrence)
    shap_values_target = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    # New format: Array (Samples, Features, Classes) -> We want all samples, all features, Class 1
    shap_values_target = shap_values[:, :, 1]
else:
    # Fallback: It might be a 2D array already
    shap_values_target = shap_values

plt.figure()
plt.title("Feature Importance: What drives Recurrence Risk?")
shap.summary_plot(shap_values_target, X_test)


# Create a dataframe of results
results = pd.DataFrame({'Actual': y_test, 'Risk_Probability': y_probs})

# Define Clinical Thresholds
# These cutoffs should be tuned based on your ROC curve, but here are standard starting points:
def stratify_risk(prob):
    if prob < 0.20:
        return 'Low Risk'
    elif prob < 0.60:
        return 'Intermediate Risk'
    else:
        return 'High Risk'

results['Risk_Group'] = results['Risk_Probability'].apply(stratify_risk)

# Validate the Groups (Does High Risk actually have more recurrences?)
print("\n--- RISK STRATIFICATION VALIDATION ---")
print(results.groupby('Risk_Group')['Actual'].mean())


# 9. SAVE THE MODEL

# This saves the trained 'rf_model' to a file named 'nsmp_risk_model.pkl'
# The file will appear in the same folder where your script is running
# 9. SAVE THE MODEL

# This saves the trained 'rf_model' to a file named 'nsmp_recurrence_model_preop.pkl'
model_filename = 'nsmp_recurrence_model_preop.pkl'

# FIX: Use SCRIPT_DIR to save the file in the 'model' directory
model_save_path = os.path.join(SCRIPT_DIR, model_filename)
joblib.dump(rf_model, model_save_path)

print(f"SUCCESS! Model saved as: {model_save_path}")
print("You can now build the calculator app.")