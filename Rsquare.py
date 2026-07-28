import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

# ==========================================
# 1. SET ACADEMIC PUBLICATION STYLE
# ==========================================
sns.set_theme(style="ticks", context="paper", font_scale=1.2)

# ==========================================
# 2. LOAD YOUR DATA
# ==========================================
target_file = '/Users/harsha/Desktop/PhD_project/Updated_Model/TestRVE1.csv'
predicted_file = '/Users/harsha/Desktop/PhD_project/Updated_Model/Combined_Master_Results.csv'

df_target = pd.read_csv(target_file)
df_predicted = pd.read_csv(predicted_file)

# ==========================================
# 3. ALIGN THE DATA
# ==========================================
df_target = df_target.rename(columns={df_target.columns[1]: 'RVE_ID'})
df_predicted = df_predicted.rename(columns={df_predicted.columns[0]: 'RVE_ID'})

df_target.columns = df_target.columns.str.strip().str.capitalize()
df_predicted.columns = df_predicted.columns.str.strip().str.capitalize()

df_target.columns = [col.upper() if col.lower().startswith('p') and col[1:].isdigit() else col for col in df_target.columns]
df_predicted.columns = [col.upper() if col.lower().startswith('p') and col[1:].isdigit() else col for col in df_predicted.columns]

df_target = df_target.rename(columns={df_target.columns[1]: 'RVE_ID'})
df_predicted = df_predicted.rename(columns={df_predicted.columns[0]: 'RVE_ID'})

df_merged = pd.merge(df_predicted, df_target, on='RVE_ID', suffixes=('_pred', '_target'))

# ==========================================
# 4. HELPER FUNCTION TO DRAW INDIVIDUAL PLOTS
# ==========================================
def create_parity_plot(ax, prop, df):
    """Handles the math and plotting for a single property to keep code clean."""
    if f'{prop}_pred' not in df.columns or f'{prop}_target' not in df.columns:
        ax.text(0.5, 0.5, f"{prop}\nMissing Data", ha='center', va='center', weight='bold', color='red')
        ax.axis('off')
        return

    x_data = pd.to_numeric(df[f'{prop}_pred'], errors='coerce')
    y_data = pd.to_numeric(df[f'{prop}_target'], errors='coerce')
    
    mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    valid_x, valid_y = x_data[mask], y_data[mask]
    
    if len(valid_x) == 0:
        return

    mae = mean_absolute_error(valid_y, valid_x)
    rmse = np.sqrt(mean_squared_error(valid_y, valid_x))
    spearman_corr, _ = spearmanr(valid_y, valid_x)
    r2 = r2_score(valid_y, valid_x)
    
    ax.scatter(valid_x, valid_y, color='royalblue', alpha=0.4, s=20, edgecolor='none')
    
    min_val = min(valid_x.min(), valid_y.min())
    max_val = max(valid_x.max(), valid_y.max())
    buffer = (max_val - min_val) * 0.05 if min_val != max_val else 0.1
    ax.set_xlim(min_val - buffer, max_val + buffer)
    ax.set_ylim(min_val - buffer, max_val + buffer)
    
    ax.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 
            color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
    
# Placed in the bottom-right corner
    ax.text(0.95, 0.05, f'R² = {r2:.3f}\nSpearman = {spearman_corr:.3f}\nMAE = {mae:.3g}\nRMSE = {rmse:.3g}', 
            transform=ax.transAxes, fontsize=11, weight='bold', 
            verticalalignment='bottom', horizontalalignment='right', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_xlabel(f'Predicted {prop}', weight='bold', fontsize=12)
    ax.set_ylabel(f'Target {prop}', weight='bold', fontsize=12)
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    
    if max_val < 0.01:
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)


# ==========================================
# 5. GENERATE THE 4 SEPARATE FIGURES
# ==========================================

# --- PLOT 1: Porosity and Mean (1x2 Grid) ---
print("Generating Plot 1: Porosity & Mean...")
fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
for i, prop in enumerate(['Porosity', 'Mean']):
    create_parity_plot(axes1[i], prop, df_merged)
fig1.tight_layout()
fig1.subplots_adjust(top=0.85)
fig1.suptitle('Accuracy: Primary Properties', fontsize=16, weight='bold')
fig1.savefig('Plot1_Porosity_Mean.png', dpi=300, bbox_inches='tight')

# --- PLOT 2: Variance, Skew, Kurtosis (1x3 Grid) ---
print("Generating Plot 2: Variance, Skew, Kurtosis...")
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
for i, prop in enumerate(['Variance', 'Skew', 'Kurtosis']):
    create_parity_plot(axes2[i], prop, df_merged)
fig2.tight_layout()
fig2.subplots_adjust(top=0.85)
fig2.suptitle('Accuracy: Higher-Order Statistical Moments', fontsize=16, weight='bold')
fig2.savefig('Plot2_Variance_Skew_Kurtosis.png', dpi=300, bbox_inches='tight')

# --- PLOT 3: Percentiles P10 - P50 (1x5 Grid) ---
print("Generating Plot 3: P10 to P50...")
p_lower = ['P10', 'P20', 'P30', 'P40', 'P50']
fig3, axes3 = plt.subplots(1, 5, figsize=(20, 4))
for i, prop in enumerate(p_lower):
    create_parity_plot(axes3[i], prop, df_merged)
fig3.tight_layout()
fig3.subplots_adjust(top=0.85)
fig3.suptitle('Accuracy: Lower Pore Size Percentiles (P10 - P50)', fontsize=16, weight='bold')
fig3.savefig('Plot3_Percentiles_Lower.png', dpi=300, bbox_inches='tight')

# --- PLOT 4: Percentiles P60 - P100 (1x5 Grid) ---
print("Generating Plot 4: P60 to P100...")
p_upper = ['P60', 'P70', 'P80', 'P90', 'P100']
fig4, axes4 = plt.subplots(1, 5, figsize=(20, 4))
for i, prop in enumerate(p_upper):
    create_parity_plot(axes4[i], prop, df_merged)
fig4.tight_layout()
fig4.subplots_adjust(top=0.85)
fig4.suptitle('Accuracy: Upper Pore Size Percentiles (P60 - P100)', fontsize=16, weight='bold')
fig4.savefig('Plot4_Percentiles_Upper.png', dpi=300, bbox_inches='tight')

print("All 4 plots generated successfully!")
plt.show()