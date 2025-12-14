import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- PORTABLE PATH SETUP ---
# Get the directory where the current script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Assume the project root is one level up (e.g., from 'data_cleaning' to 'freakyton_uterus')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 
# Define the data directory path
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# ---------------------------

# Load the translated dataset
file_path = os.path.join(DATA_DIR, "IQ_Cancer_Endometrio_merged_NMS_relevant_english.csv")
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
