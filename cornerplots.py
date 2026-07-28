import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD THE DATA
# ==========================================
file_path = '/Users/harsha/Desktop/PhD_project/Updated_Model/Generated_RVE_Parameters(400).csv'
df = pd.read_csv(file_path)

# Drop any rows with missing values in the columns we care about to avoid plotting errors
cols_to_use = ['Porosity', 'Mean', 'Variance', 'Skew', 'Kurtosis', 
               'Input1', 'Input2', 'Input3', 'Input4', 'Input5']
df = df.dropna(subset=cols_to_use)

# ==========================================
# 2. CONFIGURE PLOT SETTINGS
# ==========================================
# Define the variables for the rows (Y-axis) and columns (X-axis)
y_vars = ['Porosity', 'Mean', 'Variance', 'Skew', 'Kurtosis']
x_vars = ['Input1', 'Input2', 'Input3', 'Input4', 'Input5']

# Set the overall style
sns.set_theme(style="ticks", context="paper", font_scale=1.1)

# Create a figure and a 5x5 grid of subplots
fig, axes = plt.subplots(nrows=len(y_vars), ncols=len(x_vars), 
                         figsize=(15, 12), sharex='col', sharey='row')

# ==========================================
# 3. GENERATE THE GRID PLOTS
# ==========================================
print("Generating 2D Density Plots (this may take a moment depending on dataset size)...")

for i, y_col in enumerate(y_vars):
    for j, x_col in enumerate(x_vars):
        ax = axes[i, j]
        
        # Draw the 2D density plot
        # cmap="viridis" matches the exact yellow-to-purple color scheme in your image
        # thresh=0.05 hides the very lowest density areas to give it that "blob" look
        sns.kdeplot(
            data=df, x=x_col, y=y_col, 
            fill=True, cmap="viridis", thresh=0.05, levels=100, ax=ax
        )
        
        # Gridlines for better readability
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Formatting X-axis (Only show labels on the bottom row)
        if i == len(y_vars) - 1:
            ax.set_xlabel(x_col, weight='bold')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', bottom=False)
            
        # Formatting Y-axis (Only show labels on the leftmost column)
        if j == 0:
            ax.set_ylabel(y_col, weight='bold')
            # Use scientific notation for Variance if the values are very small
            if y_col == 'Variance':
                ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', left=False)

# ==========================================
# 4. FINALIZE AND SAVE
# ==========================================
# Adjust layout to remove space between plots (like your reference image)
plt.subplots_adjust(wspace=0.05, hspace=0.1)

# Save the high-resolution figure
output_image = '/Users/harsha/Desktop/PhD_project/Updated_Model/Density_Grid_Plot.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')

print(f"Plot successfully saved to: {output_image}")
plt.show()