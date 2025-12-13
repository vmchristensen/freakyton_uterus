import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Load data
file_path = r"C:\Users\nadine jager\Documents\hackathon\freakyton_uterus\IQ_Cancer_Endometrio_cleaned.csv"
df = pd.read_csv(file_path)

target = 'recurrence'

# 1️⃣ Explore the target
print(df[target].value_counts())
sns.countplot(x=target, data=df)
plt.title("Recurrence Distribution")
plt.show()

# 2️⃣ Identify numeric (continuous + ordinal) columns
numeric_vars = [
    'age', 'BMI', 'tumor_size', 'estrogen_receptors_pct', 
    'progesterone_receptors', 'days_to_recurrence_or_death',
    'grade', 'myometrial_invasion', 'distant_metastasis',
    'risk_group_preSurgery', 'ASA_score', 'histology_grade',
    'FIGO2023_stage', 'definitive_risk_group', 'recurrence_number'
]

# 3️⃣ Identify categorical variables
categorical_vars = [
    'histologic_type', 'neoadjuvant_treatment', 'first_surgery_treatment',
    'final_histology', 'lymph_node_involvement', 'pelvic_sentinel_nodes',
    'pelvic_lymph_nodes', 'beta_catenin', 'genetic_study_1',
    'eligible_for_radiotherapy', 'chemotherapy'
]

# Continuous variables 
continuous_vars = [ 'age', 'BMI', 'tumor_size', 
                   'estrogen_receptors_pct', 'progesterone_receptors', 
                   'days_to_recurrence_or_death' ] 
# Ordinal variables 
ordinal_vars = [ 'grade', 'myometrial_invasion', 
                'distant_metastasis', 'risk_group_preSurgery', 
                'ASA_score', 'histology_grade', 'FIGO2023_stage', 
                'definitive_risk_group', 'recurrence_number' ]
# 4️⃣ Prepare dataset
df_model = df[numeric_vars + categorical_vars + [target]].dropna()

# One-hot encode categorical variables
ohe = OneHotEncoder(sparse_output=False, drop='first')
X_cat = ohe.fit_transform(df_model[categorical_vars])
cat_cols = ohe.get_feature_names_out(categorical_vars)

# Combine numeric and categorical features
X_num = df_model[numeric_vars].values
X = np.hstack([X_num, X_cat])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df_model[target], palette='Set1')
plt.title("PCA of Patients Colored by Recurrence")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# 6️⃣ K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df_model['cluster'] = clusters

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df_model['cluster'], style=df_model[target], palette='Set2', s=100)
plt.title("Clusters and Recurrence")
plt.show()

# 7️⃣ Cluster summary
cluster_summary = df_model.groupby('cluster')[target].value_counts(normalize=True).unstack()
print("Recurrence distribution per cluster:\n", cluster_summary)






# KEEP
# Compute correlation matrix for numerical columns
corr_matrix = df[continuous_vars].corr(method='spearman' )


# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Numerical Variables")
plt.show()




