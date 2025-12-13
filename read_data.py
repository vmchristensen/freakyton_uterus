import pandas as pd

file_path = r"C:\Users\nadine jager\Documents\hackathon\freakyton_uterus\IQ_Cancer_Endometrio_merged_NMSP.xlsx"

import pandas as pd

file_path = r"C:\Users\nadine jager\Documents\hackathon\freakyton_uterus\IQ_Cancer_Endometrio_merged_NMSP.xlsx"

# Read all sheets
sheets = pd.read_excel(file_path, sheet_name=None)

# sheets is a dict: {sheet_name: DataFrame}
print(sheets.keys())

for name, df in sheets.items():
    print(f"\n{name}")
    print(df.shape)
    print(df.columns)
