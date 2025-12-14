import pandas as pd
import os

# --- PORTABLE PATH SETUP ---
# Get the directory where the current script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Assume the project root is one level up (e.g., from 'data_cleaning' to 'freakyton_uterus')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 
# Define the data directory path
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# ---------------------------


file_path = os.path.join(DATA_DIR, "IQ_Cancer_Endometrio_merged_NMSP.xlsx")

xls = pd.ExcelFile(file_path)

sheets = {}

xls = pd.ExcelFile(file_path)

# List all sheet names
print(xls.sheet_names)


for sheet_name in xls.sheet_names:
    sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)

df = sheets["IQ_Cancer_Endometrio_merged_NMS"]

print(df.head())

df.to_csv(os.path.join(DATA_DIR, "IQ_Cancer_Endometrio_merged_NMS.csv"), index=False)