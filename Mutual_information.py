import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# 1. Load the dataset
train_path = "/Users/harsha/Desktop/PhD_project/Updated_Model/Trainingdata.csv"
df = pd.read_csv(train_path, header=0)

# --- THE FIX: REMOVE THE BLANK SPACER COLUMN ---
# This line drops any column in the dataset that is entirely empty (NaN).
# It will safely delete the blank column between P100 and Input1.
df = df.dropna(axis=1, how='all')

# Just in case there are any actual missing cells in your data rows, drop those too.
original_row_count = len(df)
df = df.dropna()
if original_row_count - len(df) > 0:
    print(f"⚠️ Dropped {original_row_count - len(df)} incomplete rows.")
# -----------------------------------------------

# 2. Split the data based on your layout
# Now that the blank column is gone, Porosity is exactly 0-14, and Inputs are exactly 15-29.
porosity_df = df.iloc[:, 0:15]
solid_df = df.iloc[:, 15:30]

# Dynamically grab the exact labels from the dataframe to label the plot
porosity_labels = porosity_df.columns.tolist()
solid_labels = solid_df.columns.tolist()

# 3. Initialize an empty dataframe to hold the 15x15 MI scores
mi_matrix = pd.DataFrame(index=solid_labels, columns=porosity_labels, dtype=float)

print("Calculating Mutual Information scores... (This may take a moment)")

# 4. Loop through each solid parameter (Input1...Input15)
for i, solid_col_name in enumerate(solid_labels):
    # Extract the single solid column as the target (y)
    y = solid_df.iloc[:, i]
    
    # Calculate MI scores between all porosity features (X) and this specific solid target (y)
    mi_scores = mutual_info_regression(porosity_df, y, random_state=42)
    
    # Store the results in the matrix
    mi_matrix.loc[solid_col_name] = mi_scores

# 5. Display the numerical matrix in the console
print("\n--- Mutual Information Scores Matrix ---")
print(mi_matrix.round(3))

# 6. Visualize the matrix using a seaborn heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(
    mi_matrix, 
    annot=True,        # Show the numbers inside the squares
    fmt=".2f",         # Format to 2 decimal places
    cmap="magma_r",    # Colormap emphasizing high values
    cbar_kws={'label': 'Mutual Information Score'}
)

plt.title("Mutual Information: Porosity Features vs. Solid Microstructure (Inputs)", fontsize=16)
plt.xlabel("Porosity Phase (Conditions)", fontsize=14)
plt.ylabel("Solid Phase Microstructure (Targets)", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()