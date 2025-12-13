import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the translated dataset
file_path = r"C:\Users\nadine jager\Documents\hackathon\freakyton_uterus\IQ_Cancer_Endometrio_merged_NMS_relevant_english.csv"
df = pd.read_csv(file_path)

# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Compute correlation matrix for numerical columns
corr_matrix = df[numerical_cols].corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Numerical Variables")
plt.show()
