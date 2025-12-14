import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# --- PORTABLE PATH SETUP ---
# Get the directory where the current script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Assume the project root is one level up (e.g., from 'data_cleaning' to 'freakyton_uterus')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 
# Define the data directory path
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# ---------------------------

# Load data
file_path = os.path.join(DATA_DIR, "IQ_Cancer_Endometrio_cleaned.csv")
df = pd.read_csv(file_path)

# Target
target = 'recurrence_death'

# Pre-recurrence predictors
predictors = [
    'age', 'BMI', 'histology_grade', 'FIGO2023_stage', 'definitive_risk_group',
    'tumor_size', 'histologic_type', 'lymph_node_involvement', 'grade',
    'myometrial_invasion', 'risk_group_preSurgery', 'eligible_for_radiotherapy',
    'chemotherapy', 'estrogen_receptors_pct', 'beta_catenin', 'pelvic_lymph_nodes'
]

X = df[predictors]
y = df[target]

# Separate categorical and numeric columns
numeric_features = ['age', 'BMI', 'tumor_size', 'estrogen_receptors_pct']
categorical_features = [col for col in predictors if col not in numeric_features]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# Pipeline with Random Forest
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced'))
])

# Fit model
clf.fit(X, y)

# Get feature importances
ohe = clf.named_steps['preprocessor'].named_transformers_['cat']
cat_features_encoded = ohe.get_feature_names_out(categorical_features)
all_features = list(cat_features_encoded) + numeric_features

importances = clf.named_steps['classifier'].feature_importances_
feature_importances = pd.Series(importances, index=all_features).sort_values(ascending=False)

print("\n=== Random Forest Feature Importances (Pre-recurrence variables only) ===")
print(feature_importances)

# Plot Random Forest importances
plt.figure(figsize=(12, 8))
feature_importances.sort_values(ascending=True).plot(kind='barh', color='salmon')
plt.title("Random Forest Feature Importances (Pre-recurrence variables only)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Aggregate importances by original variable name
agg_importances = feature_importances.groupby(feature_importances.index.str.split('_').str[0]).sum().sort_values(ascending=False)
print("\n=== Aggregated Feature Importances by Original Variable ===")
print(agg_importances)

# Scale numeric features for LASSO logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LASSO logistic regression with class weighting
logreg = LogisticRegression(
    penalty='l1', 
    solver='saga', 
    max_iter=10000, 
    class_weight='balanced'
)
logreg.fit(X_scaled, y)

# Coefficients
coef = pd.Series(logreg.coef_[0], index=predictors).sort_values(key=abs, ascending=False)
print("\n=== LASSO Logistic Regression Coefficients (Class-weighted) ===")
print(coef)

# Plot LASSO coefficients
plt.figure(figsize=(10, 6))
coef.sort_values().plot(kind='barh', color='skyblue')
plt.title("Feature Importance from LASSO Logistic Regression (Class-weighted)")
plt.xlabel("Coefficient value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
