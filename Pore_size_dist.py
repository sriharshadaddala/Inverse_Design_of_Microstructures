import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Define File Paths and Target ID ---
file_name = "/Users/harsha/Desktop/PhD_project/Updated_Model/102/Pore_Results3D_102.csv"
test_file_name = "/Users/harsha/Desktop/PhD_project/Updated_Model/samplepoints.csv"

# The specific RVE_id you want to plot as the red target line
target_rve_id = 102


# --- 2. Load the CSV files ---
df_generated = pd.read_csv(file_name)
df_test = pd.read_csv(test_file_name)

# --- 3. Extract your X and Y data ---
# Get the 10 generated microstructures (assuming 10 rows in this file)
x_generated = df_generated.iloc[:, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]].values

# Find the row in Testdatamain where the 2nd column (index 1) matches the target_rve_id
# Note: If RVE_id has a specific column name like 'RVE_id', you can also use df_test[df_test['RVE_id'] == target_rve_id]
target_row = df_test[df_test.iloc[:, 1] == target_rve_id]

# Extract the exact same columns for the target data
# (Ensure columns 9-19 represent the exact same percentiles in both files)
x_target = target_row.iloc[:, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]].values

# Define your fixed Y data
y_data = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# --- 4. Set up the graph ---
plt.figure(figsize=(10, 6), dpi=150)
cmap = plt.get_cmap('tab10')
num_rves = len(x_generated)

# --- 5. Plot the Generated Data ---
for i in range(num_rves):
    plt.plot(x_generated[i], y_data, color=cmap(i), linewidth=1.5, alpha=0.7, label=f'RVE {i+1}')

# --- 6. Plot the Target Data (Red Line) ---
# Check if the target RVE was actually found to avoid errors
if len(x_target) > 0:
    # We use x_target[0] because filtering returns a 2D array, and we just want the first matching row
    plt.plot(x_target[0], y_data, color='red', linewidth=2, linestyle='--', label='Target Properties')
    print(f"Target RVE {target_rve_id} successfully plotted.")
else:
    print(f"Warning: RVE_id {target_rve_id} was not found in Testdatamain.csv")

# --- 7. Format the axes ---
plt.ylim(0, 100)      
print(f"Plotted {num_rves} Generated RVEs")

plt.title("Pore Size Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Pore Size", fontsize=12)
plt.ylabel("Cumulative Pore Percentage (%)", fontsize=12)

# --- 8. Add Legend and Grid ---
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

# --- 9. Show Plot ---
plt.tight_layout()
plt.show()