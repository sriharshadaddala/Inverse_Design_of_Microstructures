import sys
import os
import glob
import numpy as np
import pandas as pd
import imageio.v3 as iio
from skimage.filters import threshold_otsu
import porespy as ps
from porespy import metrics
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile
from scipy.stats import skew, kurtosis

# ==========================================
# 1. DEFINE YOUR FOLDERS HERE
# ==========================================
base_dir = "/Users/harsha/Desktop/PhD_project/Updated_Model"
folders_to_process = [str(i) for i in range(100, 104)] # <-- Add as many folder names as you need here

voxel_size = 0.005
save_every_n_files = 1

# Prevent plots from popping up on screen
plt.ioff()

# ==========================================
# 2. UPDATED PROCESS FUNCTION
# ==========================================
# Notice we added `current_plot_folder` as an argument
def process_image(filepath, current_plot_folder, save_plots=True):
    try:
        filename_base = os.path.basename(filepath).replace(".tif", "")
        im_raw = iio.imread(filepath)

        # 1. Standardize Geometry
        if im_raw.shape[0] == 201:
            im_raw = im_raw[:200, :200, :200]
        
        # Binarize
        if im_raw.dtype == bool:
            im = im_raw.copy()
        else:
            thresh = threshold_otsu(im_raw)
            im = im_raw < thresh

        porosity = metrics.porosity(im)

        # 2. Periodic Padding Calculation
        pad_width = 50 
        im_padded = np.pad(im, pad_width=pad_width, mode='wrap')
        lt_padded = ps.filters.local_thickness(im_padded, method="bf")
        lt = lt_padded[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]

        # 3. Save 3D TIFF (Raw Data)
        raw_map_folder = os.path.join(os.path.dirname(current_plot_folder), "RAW_MAPS")
        if not os.path.exists(raw_map_folder): 
            os.makedirs(raw_map_folder)
            
        thick_filepath = os.path.join(raw_map_folder, f"{filename_base}_thick.tif")
        tifffile.imwrite(thick_filepath, lt.astype(np.float32), compression='zlib')

        raw_pore_sizes = lt[lt > 0]
        raw_pore_sizes = raw_pore_sizes * voxel_size
        mean_ps = np.mean(raw_pore_sizes)
        var_ps = np.var(raw_pore_sizes)
        skew_ps = skew(raw_pore_sizes) 
        kurt_ps = kurtosis(raw_pore_sizes)
        
        # 5. Pore Size Distribution
        data = ps.metrics.pore_size_distribution(im=lt, bins=20, log=False, voxel_size=voxel_size)
        R, pdf, bw = data.bin_centers, data.pdf, data.bin_widths
        ps_cdf = data.cdf
        vol_fraction = pdf * bw
        R_edges = data.bin_edges
        cdf = np.insert(np.cumsum(vol_fraction), 0, 0.0)

        # Histogram Plot
        plt.figure(figsize=(10, 6))
        plt.bar(R, vol_fraction, width=bw, color='skyblue', edgecolor='black', alpha=0.8, label='Volume Fraction')
        plt.xlabel('Pore Radius(voxels)', fontsize=12)
        plt.ylabel('Volume Fraction', fontsize=12)
        plt.title('Pore Size Distribution: Volume Fraction vs Radius', fontsize=14)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(current_plot_folder, f"{filename_base}_hist.png"))
        plt.close()

        row = {
            "Filename": os.path.basename(filepath),
            "i_index": 0, "k_index": 0,
            "Porosity": round(porosity, 6),
            "Porosity_pct": round(porosity*100, 3),
            "Mean": mean_ps,
            "Variance": var_ps,
            "Skew": skew_ps,
            "Kurtosis": kurt_ps
        }
        
        # Cumulative Plot
        plt.figure(figsize=(8, 6))
        plt.plot(R_edges, cdf, marker='o', linestyle='-', color='b', label='Cumulative Distribution')
        plt.xlabel('Radius of Pores', fontsize=12)
        plt.ylabel('Cumulative Volume Fraction (0 to 1)', fontsize=12)
        plt.title('Pore Size Distribution: Cumulative Plot', fontsize=14)
        plt.ylim(0, 1.05)
        plt.xlim(left=0)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(current_plot_folder, f"{filename_base}_cdf.png"))
        plt.close()

        target_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for p in target_percentages:
            val = np.interp(p / 100.0, cdf, R_edges) 
            row[f"P{p}"] = round(val, 4)

        for b in range(len(R)):
            row[f"Radius_bin_{b+1}"] = R[b]
            row[f"VolFrac_bin_{b+1}"] = vol_fraction[b]

        return row
    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {e}")
        return None


# ==========================================
# 3. OUTER FOLDER LOOP
# ==========================================
for folder_name in folders_to_process:
    # Set up dynamic paths for this specific folder
    source_folder = os.path.join(base_dir, folder_name)
    output_csv = os.path.join(source_folder, f"Pore_Results3D_{folder_name}.csv") # Added folder name to CSV so they are distinct
    plot_folder = os.path.join(source_folder, "PLOTS")
    
    # Safety check: skip if the folder doesn't exist
    if not os.path.exists(source_folder):
        print(f"\nWarning: Folder {source_folder} not found. Skipping...")
        continue

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    print(f"\n==================================================")
    print(f"Starting Batch Processing for folder: {folder_name}")
    print(f"==================================================")

    # Reset tracking variables for the new folder
    all_results = []
    start_time = time.time()
    count = 0

    for i in range(1, 4000): 
        for k in range(1, 4):
            pattern = f"CELL_i{i:05d}_k{k:02d}_seed*.tif"
            search_path = os.path.join(source_folder, pattern)
            found_files = glob.glob(search_path)
            
            if not found_files:
                continue
                
            current_file = found_files[0]
            
            # --- PROCESS --- (Pass the plot_folder here)
            result_row = process_image(current_file, plot_folder)
            
            if result_row:
                result_row["i_index"] = i
                result_row["k_index"] = k
                all_results.append(result_row)
                count += 1

            # --- PROGRESS & SAVING TO CSV ---
            if count % save_every_n_files == 0:
                df = pd.DataFrame(all_results)
                df.to_csv(output_csv, index=False)
                
                elapsed = time.time() - start_time
                speed = count / elapsed
                print(f"[{folder_name}] Processed {count} images... ({speed:.2f} img/sec)")

    # ================== FINAL SAVE FOR THIS FOLDER ==================
    if all_results:
        print(f"Loop finished for {folder_name}. Saving final file...")
        df_final = pd.DataFrame(all_results)
        df_final.to_csv(output_csv, index=False)
        print(f"Successfully saved {len(df_final)} rows to: {output_csv}")
        print(f"Plots saved to: {plot_folder}")
    else:
        print(f"No images found or processed in folder {folder_name}.")

print("\n🎉 All folders processed successfully!")