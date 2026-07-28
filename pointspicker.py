import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Mute plotting warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD DATA
# ==========================================
sns.set_theme(style="ticks", context="paper", font_scale=1.0)
file_name = "/Users/harsha/Desktop/PhD_project/Updated_Model/Trainingdata.csv"
cols = ['Porosity', 'Mean', 'Variance', 'Skew', 'Kurtosis', 'P10', 'P20', 'P30', 'P40', 'P50', 'P60', 'P70', 'P80', 'P90', 'P100']
df = pd.read_csv(file_name, usecols=cols).dropna()

# ==========================================
# 2. GENERATE 3 NOVEL SAMPLES
# ==========================================
print("Generating 3 novel RVE samples...")
valid_samples = []
while len(valid_samples) < 3:
    base_point = df.sample(1).values.flatten()
    new_point = base_point * np.random.uniform(0.985, 1.015, size=15)
    
    poro, mean, var = new_point[:3]
    if poro > 0 and var > 0 and np.all(np.diff(new_point[5:]) >= 0):
        valid_samples.append(new_point)

df_results = pd.DataFrame(valid_samples, columns=cols)
print("\n--- GENERATED TEST SAMPLES ---")
print(df_results.to_string())

# --- ADD THIS NEW LINE RIGHT HERE ---
df_results.to_csv("/Users/harsha/Desktop/PhD_project/Updated_Model/Generated_point_Samples.csv", index=False)

# ==========================================
# 3. REUSABLE PLOTTING FUNCTION
# ==========================================

# ==========================================
# 3. REUSABLE PLOTTING FUNCTION
# ==========================================
def safe_kdeplot(x, y, **kwargs):
    """Tries to draw smooth curves; falls back to histograms if math crashes."""
    try:
        sns.kdeplot(x=x, y=y, **kwargs)
    except (ValueError, np.linalg.LinAlgError):
        sns.histplot(x=x, y=y, bins=100, cmap=kwargs.get('cmap', 'viridis'), pmax=0.9)

def create_and_show_plot(cols_to_plot, title):
    """Builds a 5x5 grid for a specific set of columns and pauses the script until closed."""
    print(f"\nDrawing: {title}...")
    
    df_subset = df[cols_to_plot]
    g = sns.PairGrid(df_subset, corner=True, diag_sharey=False)

    g.map_lower(safe_kdeplot, fill=True, cmap='viridis', warn_singular=False, 
                levels=20, thresh=0.05, cut=0, gridsize=400, bw_adjust=1)

    for i, j in zip(*np.triu_indices_from(g.axes, k=0)):
        if g.axes[i, j] is not None:
            g.axes[i, j].set_visible(False)

    sample_styles = [
        {'color': 'red', 'marker': '*', 's': 250, 'label': 'S1 '},
        {'color': 'cyan', 'marker': 'D', 's': 100, 'label': 'S2 '},
        {'color': 'magenta', 'marker': '^', 's': 150, 'label': 'S3'}
    ]

    for i in range(len(cols_to_plot)):
        for j in range(len(cols_to_plot)):
            ax = g.axes[i, j]
            if ax is not None and i > j:
                x_col = cols_to_plot[j]
                y_col = cols_to_plot[i]
                
                # Overlay points
                for idx, style in enumerate(sample_styles):
                    ax.scatter(df_results.iloc[idx][x_col], df_results.iloc[idx][y_col], 
                               color=style['color'], marker=style['marker'], s=style['s'], 
                               edgecolor='black', linewidths=1.5, zorder=10, 
                               label=style['label'] if (i==1 and j==0) else "") 

                # Clamp axes tightly to data range
                x_min, x_max = df[x_col].quantile(0.01), df[x_col].quantile(0.99)
                y_min, y_max = df[y_col].quantile(0.01), df[y_col].quantile(0.99)
                
                x_min, x_max = min(x_min, df_results[x_col].min()), max(x_max, df_results[x_col].max())
                y_min, y_max = min(y_min, df_results[y_col].min()), max(y_max, df_results[y_col].max())
                
                x_buffer, y_buffer = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
                ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
                ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    g.figure.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85), fontsize=12, markerscale=1, frameon=True, shadow=True)
    g.figure.suptitle(title, weight='bold', fontsize=16, y=0.98) # Add a title to the top
    
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    
    # This pauses the script. It will not continue until you close the window!
    plt.show() 

# ==========================================
# 4. EXECUTE THE 3 PLOTS SEQUENTIALLY
# ==========================================
group_1 = ['Porosity', 'Mean', 'Variance', 'Skew', 'Kurtosis']
group_2 = ['P10', 'P20', 'P30', 'P40', 'P50']
group_3 = ['P60', 'P70', 'P80', 'P90', 'P100']

# Pop-up 1
create_and_show_plot(group_1, "Plot 1: Primary Properties")

# Pop-up 2 (Appears after you close Plot 1)
create_and_show_plot(group_2, "Plot 2: Lower Percentiles (P10 - P50)")

# Pop-up 3 (Appears after you close Plot 2)
create_and_show_plot(group_3, "Plot 3: Upper Percentiles (P60 - P100)")

print("\nAll plots generated successfully!")