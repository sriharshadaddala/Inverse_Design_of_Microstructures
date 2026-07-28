import pandas as pd
import numpy as np

# 1. Load your REAL training data
TRAIN_DATA_PATH = "/Users/harsha/Desktop/PhD_project/Updated_Model/Trainingdata.csv"
df_real = pd.read_csv(TRAIN_DATA_PATH)

print(f"Loaded real data: {len(df_real)} samples")

# 2. Isolate the inputs (the 15 parameters the GAN generates)
# Make sure these column indices match your actual data layout!
# Usually 0:15 are conditions, column 15 is the ghost 'P', 16:31 are inputs
conditions = df_real.iloc[:, 0:15].copy()
inputs = df_real.iloc[:, 16:31].copy() 

# 3. The "Shuffling" Magic
shuffled_inputs = inputs.copy()
for col in shuffled_inputs.columns:
    # We independently shuffle every single column to destroy 2D correlations
    # while perfectly preserving the 1D probability density of each parameter
    shuffled_inputs[col] = np.random.permutation(shuffled_inputs[col].values)

# 4. Recombine with the untouched real conditions
# We leave conditions intact, exactly like we did for the uniform baseline
df_shuffled_baseline = pd.concat([conditions, shuffled_inputs], axis=1)

# 5. Save it to test in your JS script
OUTPUT_PATH = "/Users/harsha/Desktop/PhD_project/Updated_Model/Shuffled_Baseline.csv"
df_shuffled_baseline.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Shuffled Baseline saved to: {OUTPUT_PATH}")