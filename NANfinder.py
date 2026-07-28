import pandas as pd
import numpy as np

# Load just the 5 columns you are trying to plot
file_name = "/Users/harsha/Desktop/PhD_project/Updated_Model/Trainingdata.csv"
cols_to_use = ['Porosity', 'Mean', 'Variance', 'Skew', 'Kurtosis']
df = pd.read_csv(file_name, usecols=cols_to_use)

print("--- DIAGNOSTIC REPORT ---")

# 1. Check for Text / Non-Numeric Junk
for col in df.columns:
    # Try converting to numbers
    converted = pd.to_numeric(df[col], errors='coerce')
    # Find rows that failed conversion
    bad_rows = df[converted.isna() & df[col].notna()]
    
    if not bad_rows.empty:
        print(f"\n❌ ERROR: Found text/junk values in column: '{col}'")
        print("Here are the first few problematic values:")
        # Print the exact text causing the issue
        print(bad_rows[col].head()) 

# 2. Check for Zero Variance (Constant values)
for col in df.columns:
    # If there is only 1 unique value in the whole column
    if df[col].nunique() <= 1:
        val = df[col].iloc[0]
        print(f"\n❌ ERROR: Column '{col}' has zero variance!")
        print(f"Every single row is identical (Value: {val}). KDE cannot plot this.")

# 3. Check for NaNs or Infinities
for col in df.columns:
    nan_count = df[col].isna().sum()
    inf_count = np.isinf(pd.to_numeric(df[col], errors='coerce')).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"\n⚠️ WARNING: '{col}' has {nan_count} missing values (NaN) and {inf_count} infinite values.")

print("\n--- DIAGNOSTICS COMPLETE ---")