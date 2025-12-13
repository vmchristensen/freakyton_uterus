import pandas as pd

file_path = r"C:\Users\nadine jager\Documents\hackathon\freakyton_uterus\IQ_Cancer_Endometrio_merged_NMSP.xlsx"

xls = pd.ExcelFile(file_path)

sheets = {}

xls = pd.ExcelFile(file_path)

# List all sheet names
print(xls.sheet_names)


for sheet_name in xls.sheet_names:
    sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)

df = sheets["IQ_Cancer_Endometrio_merged_NMS"]

print(df.head())