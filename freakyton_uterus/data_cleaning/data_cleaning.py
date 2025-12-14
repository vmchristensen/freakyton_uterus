import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from scipy.stats import skew
import os

# --- PORTABLE PATH SETUP ---
# Get the directory where the current script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Assume the project root is one level up (e.g., from 'data_cleaning' to 'freakyton_uterus')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 
# Define the data directory path
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# ---------------------------


# -------------------------------
# Load dataset
# -------------------------------
file_path = os.path.join(DATA_DIR, "IQ_Cancer_Endometrio_merged_NMS_relevant_english.csv")
df = pd.read_csv(file_path)

# -------------------------------
# Define comprehensive variable lists based on data type and meaning
# -------------------------------
# Continuous: Measures that can take any value in a range
continuous_vars = [
    'age', 'BMI', 'tumor_size', 'CA125_value', 
    'estrogen_receptors_pct', 'progesterone_receptors',
    'days_to_recurrence_or_death'
]

# Ordinal: Ordered categories (e.g., grades, stages)
ordinal_vars = [
    'grade', 'myometrial_invasion', 'risk_group_preSurgery', 'ASA_score', 
    'histology_grade', 'FIGO2023_stage', 'definitive_risk_group', 
    'recurrence_number', 'pelvic_sentinel_nodes', 'pelvic_lymph_nodes',
    'para_aortic_nodes'
]

# Categorical/Binary (All Encoded as Numeric): Discrete categories for mode imputation
categorical_vars = [
    'distant_metastasis', 'neoadjuvant_treatment', 'lymph_node_involvement', 
    'beta_catenin', 'genetic_study_1', 'eligible_for_radiotherapy', 
    'radiotherapy_treatment', 'systemic_treatment', 'recurrence_death', 
    'cause_of_death', 'disease_free', 'complete_macroscopic_resection', 
    'chemotherapy', 'first_surgery_treatment', 'histologic_type', 
    'final_histology', 'recurrence_dx', 'recurrence_treatment', 
    'recurrence_surgery', 'current_status', 'recurrence'
]

# Date/Time variables (will be checked but generally not imputed with mean/median/mode)
date_vars = [
    'birth_date', 'diagnosis_date', 'last_visit_date', 
    'death_date', 'recurrence_date'
]


# -------------------------------
# Missing data helper function
# -------------------------------
def missing_info(df, cols):
    # Filter to only columns that exist in the current DataFrame
    cols = [c for c in cols if c in df.columns] 
    if not cols:
        return pd.DataFrame()
        
    missing_count = df[cols].isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_table = pd.DataFrame({
        'missing_count': missing_count,
        'missing_pct': missing_pct
    })
    return missing_table[missing_table['missing_count'] > 0].sort_values('missing_pct', ascending=False)

print("Initial missing CONTINUOUS values:\n", missing_info(df, continuous_vars))
print("\nInitial missing ORDINAL values:\n", missing_info(df, ordinal_vars))
print("\nInitial missing CATEGORICAL/BINARY values:\n", missing_info(df, categorical_vars))
print("\nInitial missing DATE/OBJECT values:\n", missing_info(df, date_vars))


# -------------------------------
# Drop rows with unknown recurrence (recurrence == 2)
# -------------------------------
if 'recurrence' in df.columns:
    # Ensure recurrence column is numeric for comparison
    df['recurrence'] = pd.to_numeric(df['recurrence'], errors='coerce') 
    unknown_rec_count = (df['recurrence'] == 2).sum()
    df = df[df['recurrence'] != 2]
    print(f"\nDropped {unknown_rec_count} rows with unknown recurrence (recurrence == 2)")

# -------------------------------
# Convert date columns to datetime
# -------------------------------

# Fix specific known incorrect date entry
df.loc[df['birth_date'] == '05/60/1961', 'birth_date'] = '05/06/1961'


# Columns with DD/MM/YYYY format
ddmmyyyy_cols = ['birth_date', 'death_date', 'recurrence_date']

for col in ddmmyyyy_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

# Columns with YYYY-MM-DD format
iso_cols = ['diagnosis_date', 'last_visit_date']

for col in iso_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# -------------------------------
# Recurrence categorical cleanup
# -------------------------------
if 'recurrence_number' in df.columns:
    df.loc[df['recurrence'] == 0, 'recurrence_number'] = 0
recurrence_cats = ['recurrence_surgery', 'recurrence_treatment', 'recurrence_dx', 'complete_macroscopic_resection']
for col in recurrence_cats:
    if col in df.columns:
        df.loc[df['recurrence'] == 0, col] = -1 

# -------------------------------
# Systemic treatment cleanup
# -------------------------------
if 'chemotherapy' in df.columns and 'systemic_treatment' in df.columns:
    df.loc[df['chemotherapy'] == 0, 'systemic_treatment'] = -1


# -------------------------------
# Correct unrealistic ages (<30)
# -------------------------------
if 'age' in df.columns:
    median_age = df.loc[df['age'] >= 30, 'age'].median()
    mask_wrong_age = df['age'] < 30
    df.loc[mask_wrong_age, 'age'] = median_age
    print(f"Corrected {mask_wrong_age.sum()} unrealistic ages to median age {median_age}")

# -------------------------------
# Compute diagnosis date from birth date + age
# -------------------------------
if 'birth_date' in df.columns and 'age' in df.columns:
    mask_diag_from_birth = df['diagnosis_date'].isna() & df['birth_date'].notna() & df['age'].notna()
    df.loc[mask_diag_from_birth, 'diagnosis_date'] = df.loc[mask_diag_from_birth, 'birth_date'] + pd.to_timedelta(df.loc[mask_diag_from_birth, 'age'] * 365.25, unit='D')
    print(f"Computed diagnosis_date from birth_date + age for {mask_diag_from_birth.sum()} rows")

# -------------------------------
# Recurrence and death date calculation
# -------------------------------
if 'recurrence_date' in df.columns and 'days_to_recurrence_or_death' in df.columns:
    mask_rec = (df['recurrence'] == 1) & df['recurrence_date'].isna() & df['days_to_recurrence_or_death'].notna()
    df.loc[mask_rec, 'recurrence_date'] = df.loc[mask_rec, 'diagnosis_date'] + pd.to_timedelta(df.loc[mask_rec, 'days_to_recurrence_or_death'], unit='D')
    df.loc[df['recurrence'] == 0, 'recurrence_date'] = pd.NaT

if 'death_date' in df.columns and 'days_to_recurrence_or_death' in df.columns:
    mask_death = (df['recurrence'] == 0) & (df['current_status'] == 2) & df['death_date'].isna() & df['days_to_recurrence_or_death'].notna()
    df.loc[mask_death, 'death_date'] = df.loc[mask_death, 'diagnosis_date'] + pd.to_timedelta(df.loc[mask_death, 'days_to_recurrence_or_death'], unit='D')


# -------------------------------
# Second detailed missing value check
# -------------------------------
print("\nSecond missing CONTINUOUS values:\n", missing_info(df, continuous_vars))
print("\nSecond missing ORDINAL values:\n", missing_info(df, ordinal_vars))
print("\nSecond missing CATEGORICAL/BINARY values:\n", missing_info(df, categorical_vars))
print("\nSecond missing DATE/OBJECT values:\n", missing_info(df, date_vars))


# -------------------------------
# Drop columns above 70% missing
# -------------------------------
threshold = 70
all_numerical_cols = continuous_vars + ordinal_vars + categorical_vars
num_missing = missing_info(df, all_numerical_cols)
num_to_drop = num_missing[num_missing['missing_pct'] > threshold].index.tolist()
df.drop(columns=num_to_drop, inplace=True, errors='ignore')
print(f"\nDropped columns above {threshold}% missing: {num_to_drop}")

# Update lists by removing dropped columns
continuous_vars = [c for c in continuous_vars if c not in num_to_drop]
ordinal_vars = [c for c in ordinal_vars if c not in num_to_drop]
categorical_vars = [c for c in categorical_vars if c not in num_to_drop]


# -------------------------------
# Improved imputation block (Regression + Mean/Median/Mode)
# -------------------------------

corr_threshold = 0.25 # minimum correlation for regression
# Only use Continuous and Ordinal variables for regression targets
regression_targets = [v for v in (continuous_vars + ordinal_vars) if v in df.columns]

# Compute correlation matrix once
corr_matrix = df[[c for c in regression_targets if c in df.columns]].corr()

for col in regression_targets:
    n_missing = df[col].isna().sum()
    if n_missing == 0:
        continue

    # --- Regression Imputation Attempt ---
    target_corr = corr_matrix[col].dropna().sort_values(ascending=False)
    predictors = target_corr[target_corr >= corr_threshold].index.tolist()
    if col in predictors:
        predictors.remove(col)

    if predictors:
        # Prepare training data
        train = df[df[col].notna()]
        X_train = train[predictors]
        y_train = train[col]

        # Drop rows with missing predictors
        valid_idx = X_train.dropna().index
        X_train = X_train.loc[valid_idx]
        y_train = y_train.loc[valid_idx]

        # Prepare missing rows
        missing_mask = df[col].isna()
        X_missing = df.loc[missing_mask, predictors]
        X_missing = X_missing.fillna(X_train.median()) 

        if not X_train.empty and not X_missing.empty and len(X_train) >= 2:
            reg = LinearRegression()
            try:
                reg.fit(X_train, y_train)
                preds = reg.predict(X_missing)

                # Post-processing
                if col in ordinal_vars:
                    preds = np.round(preds)
                    valid_min = df[col].dropna().min()
                    valid_max = df[col].dropna().max()
                    preds = np.clip(preds, valid_min, valid_max)
                    preds = preds.astype(int) 
                elif col in continuous_vars:
                    # Prevent negative predictions for physical/continuous quantities
                    preds[preds < 0] = np.nan 

                df.loc[missing_mask, col] = preds
                print(f"Imputed {n_missing} missing values for '{col}' using regression on: {predictors}")
                continue
            except Exception as e:
                pass # Fallback to mean/median/mode below

    # --- Fallback Imputation (Mean/Median for Continuous, Mode for Ordinal) ---
    missing_mask = df[col].isna()
    if missing_mask.sum() > 0:
        col_data = df[col].dropna()
        col_skew = skew(col_data) if len(col_data) > 2 else 0
        
        if col in continuous_vars:
            if abs(col_skew) > 1:
                fill_value = df[col].median()
                method = 'median'
            else:
                fill_value = df[col].mean()
                method = 'mean'
        else: # Ordinal variables fallback to mode/median
            fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else df[col].median()
            method = 'mode'

        df.loc[missing_mask, col] = fill_value
        print(f"Imputed {missing_mask.sum()} missing values for '{col}' using {method} (fallback)")

# -------------------------------
# Impute remaining CATEGORICAL/BINARY columns using Mode
# This final block ensures all encoded numeric variables have no NaNs.
# -------------------------------
all_categorical_to_fill = [v for v in categorical_vars if v in df.columns]

for col in all_categorical_to_fill:
    missing_mask = df[col].isna()
    n_missing = missing_mask.sum()
    
    if n_missing > 0:
        try:
            # Use mode imputation for all categorical and binary columns
            fill_value = df[col].mode()[0]
            df.loc[missing_mask, col] = fill_value
            print(f"Imputed {n_missing} missing values for CATEGORICAL/BINARY '{col}' using MODE.")
        except IndexError:
            print(f"WARNING: Could not impute {col} using MODE (no valid values found).")


# -------------------------------
# Final checks
# -------------------------------
print("\nFINAL missing CONTINUOUS values:\n", missing_info(df, continuous_vars))
print("\nFINAL missing ORDINAL values:\n", missing_info(df, ordinal_vars))
print("\nFINAL missing CATEGORICAL/BINARY values:\n", missing_info(df, categorical_vars))
print("\nFINAL missing DATE/OBJECT values:\n", missing_info(df, date_vars))

# -------------------------------
# Save final cleaned DataFrame
# -------------------------------
output_path = os.path.join(DATA_DIR, "IQ_Cancer_Endometrio_cleaned.csv")
df.to_csv(output_path, index=False)
print(f"\nFinal cleaned DataFrame saved to: {output_path}")