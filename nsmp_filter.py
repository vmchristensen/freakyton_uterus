import pandas as pd
import numpy as np

# Load CSV
file_path = r"C:\Users\nadine jager\Documents\hackathon\freakyton_uterus\IQ_Cancer_Endometrio_merged_NMS.csv"
df = pd.read_csv(file_path)

abnormal_mask = (
    (df['mut_pole'] == 1) |
    (df['p53_molecular'] == 1) |
    (df['p53_ihq'] == 2) |
    (df['msh2'] == 1) |
    (df['msh6'] == 1) |
    (df['pms2'] == 1) |
    (df['mlh1'] == 1)
)

# Set NSMP = 0 if any abnormal
df['NSMP'] = np.where(abnormal_mask, 0, np.nan)

# Check results
print(df['NSMP'].value_counts(dropna=False))

