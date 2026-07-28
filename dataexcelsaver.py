import os
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
base_dir = "/Users/harsha/Library/CloudStorage/GoogleDrive-sreeharshadaddala@gmail.com/My Drive/NEW_RVE_SET/Test_Results_400voxel"
output_filename = "Combined_Master_Results.csv"
output_path = os.path.join(base_dir, output_filename)

# Define the folder range (2980 to 3100 inclusive)
start_folder = 3350
end_folder = 3501

# List to hold all the merged dataframes before final compilation
all_combined_data = []

# ==========================================
# 2. PROCESSING LOOP
# ==========================================
print("Starting data consolidation...")

for folder_id in range(start_folder, end_folder + 1):
    folder_name = str(folder_id)
    folder_path = os.path.join(base_dir, folder_name)
    
    # Paths to the two target files
    param_file = os.path.join(folder_path, f"{folder_name}.csv")
    pore_file = os.path.join(folder_path, f"Pore_Results3D_{folder_name}.csv")
    
    # Check if BOTH files exist before trying to read them
    if os.path.exists(param_file) and os.path.exists(pore_file):
        try:
            # Read the CSVs
            df_param = pd.read_csv(param_file, nrows=10)
            df_pore = pd.read_csv(pore_file)
            
            # Combine the two dataframes side-by-side (merging columns)
            df_merged = pd.concat([df_param, df_pore], axis=1)
            
            # Insert the Folder ID as the first column for tracking
            df_merged.insert(0, 'RVE_ID', folder_name)
            
            # Append to our master list
            all_combined_data.append(df_merged)
            
            print(f"✅ Processed folder {folder_name}")
            
        except Exception as e:
            print(f"❌ Error reading files in folder {folder_name}: {e}")
    else:
        print(f"⚠️ Missing files in folder {folder_name}. Skipping...")

# ==========================================
# 3. EXPORT FINAL MASTER FILE
# ==========================================
if all_combined_data:
    # Stack all the individual folder dataframes on top of each other
    final_master_df = pd.concat(all_combined_data, ignore_index=True)
    
    # Save to a single CSV file
    final_master_df.to_csv(output_path, index=False)
    print("\n==================================================")
    print(f"🎉 Success! Combined data saved to: {output_path}")
    print(f"Total rows collected: {len(final_master_df)}")
    print("==================================================")
else:
    print("\n❌ No data was found or processed. Please check your file paths.")