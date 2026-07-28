import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_diagnostic_heatmap(csv_path):
    """
    Reads 2D KL Divergence data and generates a diagnostic heatmap.
    """
    print(f"Loading data from {csv_path}...")
    
    # 1. Load the data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_path}.")
        return

    # 2. Prepare the Matrix (The Bulletproof Method)
    # By using the column's index position [0, 1, 2], we completely ignore 
    # any typos, hidden spaces, or weird formatting in the CSV headers.
    col_y = df.columns[0]   # The 'Quantifying_I...' column
    col_x = df.columns[1]   # The 'Input_Param' column
    col_val = df.columns[2] # The '2D_KL_Score' column

    print(f"Processing matrix using columns: '{col_y}', '{col_x}', and '{col_val}'")

    try:
        heatmap_data = df.pivot(index=col_y, columns=col_x, values=col_val)
    except Exception as e:
        print(f"Matrix conversion failed. Error: {e}")
        return

    # 3. Set up the visual canvas
    plt.figure(figsize=(14, 12)) # Generous sizing for a 15x15 grid
    sns.set_theme(style="white")

    # 4. Generate the Heatmap
    ax = sns.heatmap(
        heatmap_data, 
        cmap="coolwarm", 
        annot=True,          # Prints the exact KL score inside the box
        fmt=".2f",           # Rounds to 2 decimal places
        annot_kws={"size": 8}, # Font size for the internal numbers
        linewidths=0.5,      # Clear gridlines
        linecolor='white',
        cbar_kws={'label': 'KL Divergence Score (Lower is Better)'}
    )

    # 5. Professional Formatting
    plt.title('GAN Performance Diagnostic: 2D Pairwise KL Divergence', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Input Parameters', fontsize=14, fontweight='bold')
    plt.ylabel('Quantifying Index (RVEs)', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 6. Save and Display
    output_filename = '/Users/harsha/Desktop/PhD_project/Updated_Model/GAN_Diagnostic_Heatmap.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight') 
    
    print(f"\nSuccess! Heatmap saved directly to:")
    print(output_filename)
    
    plt.show()

# Run the function pointing directly to your exact file path
if __name__ == "__main__":
    target_csv = '/Users/harsha/Desktop/PhD_project/Updated_Model/2D_KL_Divergence_Report.csv'
    generate_diagnostic_heatmap(target_csv)