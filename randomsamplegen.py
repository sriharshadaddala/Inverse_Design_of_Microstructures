import pandas as pd
import numpy as np
import os

# 1. Configuration
train_path = "/Users/harsha/Desktop/PhD_project/Updated_Model/Generated_RVE_Parameters(400).csv"
save_path = "/Users/harsha/Desktop/PhD_project/Updated_Model/Random_Baseline1.csv"

condition_cols = ['Porosity', 'Mean', 'Variance', 'Skew', 'Kurtosis', 'P10', 'P20', 'P30', 'P40', 'P50', 
                  'P60', 'P70', 'P80', 'P90', 'P100']
input_cols = [f'Input{i}' for i in range(1, 16)] # Input1 to Input15

# 2. Load the training data
print("Loading training data to create conditional baseline...")
df_train = pd.read_csv(train_path)

# Create a copy to act as our baseline
df_baseline = df_train.copy()

# 3. Generate Random Baseline (Only for Input Columns)
# We keep conditions fixed, we only randomize the 15 input features
print("Generating random inputs while preserving conditions...")

for col in input_cols:
    # Use the min/max of the original training data for this specific column
    low = df_train[col].min()
    high = df_train[col].max()
    
    # Generate random uniform noise within the physical bounds
    df_baseline[col] = np.random.uniform(low, high, size=len(df_train))

# 4. Save the Conditional Baseline
df_baseline.to_csv(save_path, index=False)
print(f"✅ Conditional random baseline saved to: {save_path}")