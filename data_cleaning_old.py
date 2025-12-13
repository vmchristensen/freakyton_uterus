import pandas as pd
from sklearn.impute import SimpleImputer

# -------------------------------
# Load dataset
# -------------------------------
file_path = r"C:\Users\nadine jager\Documents\hackathon\freakyton_uterus\IQ_Cancer_Endometrio_merged_NMS_relevant_english.csv"
df = pd.read_csv(file_path)


# -------------------------------
# Drop rows with unknown recurrence (recurrence == 2)
# -------------------------------
if 'recurrence' in df.columns:
    unknown_rec_count = (df['recurrence'] == 2).sum()
    df = df[df['recurrence'] != 2]
    print(f"Dropped {unknown_rec_count} rows with unknown recurrence (recurrence == 2)")

# -------------------------------
# Convert date columns to datetime
# -------------------------------
date_cols = ['diagnosis_date', 'recurrence_date', 'death_date', 'last_visit_date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# -------------------------------
# Identify numerical and categorical columns
# -------------------------------
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'bool', 'datetime64[ns]']).columns.tolist()

# -------------------------------
# Missing data helper function
# -------------------------------
def missing_info(df, cols):
    missing_count = df[cols].isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_table = pd.DataFrame({
        'missing_count': missing_count,
        'missing_pct': missing_pct
    })
    return missing_table[missing_table['missing_count'] > 0].sort_values('missing_pct', ascending=False)

print("Initial missing numerical values:\n", missing_info(df, numerical_cols))
print("\nInitial missing categorical values:\n", missing_info(df, categorical_cols))

# -------------------------------
# Drop numerical columns above 70% missing
# -------------------------------
threshold = 70
num_missing = missing_info(df, numerical_cols)
num_to_drop = num_missing[num_missing['missing_pct'] > threshold].index.tolist()
df.drop(columns=num_to_drop, inplace=True)
print(f"\nDropped numerical columns above {threshold}% missing:\n{num_to_drop}")

# Update numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# -------------------------------
# Median imputation for remaining numeric columns
# -------------------------------
num_missing = missing_info(df, numerical_cols)
num_to_impute = num_missing[num_missing['missing_pct'] < threshold].index.tolist()
imputer = SimpleImputer(strategy='median')
df[num_to_impute] = imputer.fit_transform(df[num_to_impute])

# -------------------------------
# Recurrence categorical cleanup
# -------------------------------
df.loc[df['recurrence'] == 0, 'recurrence_number'] = 0
recurrence_cats = ['recurrence_surgery', 'recurrence_treatment', 'recurrence_dx', 'complete_macroscopic_resection']
for col in recurrence_cats:
    if col in df.columns:
        df.loc[df['recurrence'] == 0, col] = -1



# -------------------------------
# Diagnosis date imputation
# -------------------------------
if 'diagnosis_date' in df.columns:
    valid_diag = df['diagnosis_date'].dropna()
    if not valid_diag.empty:
        most_frequent_diag = valid_diag.mode().iloc[0]
        mask_diag = df['diagnosis_date'].isna()
        df.loc[mask_diag, 'diagnosis_date'] = most_frequent_diag
        print(f"\nDiagnosis_date imputed for {mask_diag.sum()} rows using {most_frequent_diag.date()}")

# -------------------------------
# Birth date calculation: diagnosis_date - age
# -------------------------------
if 'age' in df.columns:
    mask_birth = df.get('birth_date', pd.Series([pd.NaT]*len(df))).isna() & df['diagnosis_date'].notna()
    df.loc[mask_birth, 'birth_date'] = df.loc[mask_birth, 'diagnosis_date'] - pd.to_timedelta(df.loc[mask_birth, 'age'] * 365.25, unit='D')
    print(f"Birth date calculated for {mask_birth.sum()} rows using diagnosis_date - age")

# -------------------------------
# Recurrence date calculation
# -------------------------------
if 'recurrence_date' in df.columns and 'days_to_recurrence_or_death' in df.columns:
    mask_rec = (df['recurrence'] == 1) & df['recurrence_date'].isna() & df['days_to_recurrence_or_death'].notna()
    df.loc[mask_rec, 'recurrence_date'] = df.loc[mask_rec, 'diagnosis_date'] + pd.to_timedelta(df.loc[mask_rec, 'days_to_recurrence_or_death'], unit='D')
    df.loc[df['recurrence'] == 0, 'recurrence_date'] = pd.NaT

# -------------------------------
# Death date calculation
# -------------------------------
if 'death_date' in df.columns and 'days_to_recurrence_or_death' in df.columns:
    mask_death = (df['recurrence'] == 0) & df['death_date'].isna() & df['days_to_recurrence_or_death'].notna()
    df.loc[mask_death, 'death_date'] = df.loc[mask_death, 'diagnosis_date'] + pd.to_timedelta(df.loc[mask_death, 'days_to_recurrence_or_death'], unit='D')

# -------------------------------
# Final checks
# -------------------------------
if 'last_visit_date' in df.columns:
    invalid_rec = df[df['recurrence_date'] > df['last_visit_date']]
    print(f"Potentially invalid recurrence dates after last_visit_date: {len(invalid_rec)}")

# -------------------------------
# Final missing data check
# -------------------------------
categorical_cols = df.select_dtypes(include=['object', 'bool', 'datetime64[ns]']).columns.tolist()
print("\nFINAL missing numerical values:\n", missing_info(df, numerical_cols))
print("\nFINAL missing categorical values:\n", missing_info(df, categorical_cols))

# -------------------------------
# Save final cleaned DataFrame
# -------------------------------
output_path = r"C:\Users\nadine jager\Documents\hackathon\freakyton_uterus\IQ_Cancer_Endometrio_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"\nFinal cleaned DataFrame saved to: {output_path}")