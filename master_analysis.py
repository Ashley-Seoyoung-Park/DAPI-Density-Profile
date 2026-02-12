
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import interp1d

def run_master_analysis():
    # 1. Select Folder
    root = tk.Tk(); root.withdraw()
    target_dir = filedialog.askdirectory(title="Select the parent folder containing analysis results")
    if not target_dir: return

    # 2. Collect File List
    profile_files = glob.glob(os.path.join(target_dir, "**", "*_profile.csv"), recursive=True)
    summary_files = glob.glob(os.path.join(target_dir, "**", "*_summary.csv"), recursive=True)

    # Exclude the master aggregated files themselves to prevent recursion/duplication
    profile_files = [f for f in profile_files if "Master_Aggregated_Profile.csv" not in os.path.basename(f)]
    summary_files = [f for f in summary_files if "Master_Aggregated_Summary.csv" not in os.path.basename(f)]

    if not profile_files:
        print("No profile.csv files found to analyze.")
        return

    print(f"Aggregating analysis for {len(profile_files)} samples.")

    # 3. Aggregate and Interpolate Profile Data
    # v13: Limit to 256 points for Prism compatibility
    MAX_POINTS = 256
    x_master = np.linspace(-1.5, 1.5, MAX_POINTS)
    all_profiles = []
    profile_names = []

    for f in profile_files:
        df = pd.read_csv(f)
        fname = os.path.basename(f).replace("_profile.csv", "")
        
        # v13 Compatibility: Automatic column name detection
        available_cols = df.columns.tolist()
        
        # Find Intensity Column
        if 'Intensity' in available_cols:
            y_col = 'Intensity'
        elif 'Intensity_Smoothed' in available_cols:
            y_col = 'Intensity_Smoothed'
        elif 'Mean_Intensity' in available_cols:
            y_col = 'Mean_Intensity'
        else:
            print(f"Warning: Cannot find intensity column in {fname}, available: {available_cols}")
            continue
            
        # Find Position Column
        if 'Pos_Norm' in available_cols:
            x_col = 'Pos_Norm'
        elif 'Pos_Normalized' in available_cols:
            x_col = 'Pos_Normalized'
        else:
            print(f"Warning: Cannot find position column in {fname}, available: {available_cols}")
            continue
        
        y_vals = np.clip(df[y_col].values, 0, None)
        x_vals = df[x_col].values
        
        # Perform Interpolation
        f_interp = interp1d(x_vals, y_vals, bounds_error=False, fill_value=0)
        all_profiles.append(f_interp(x_master))
        profile_names.append(fname)

    all_profiles = np.array(all_profiles)
    mean_profile = np.mean(all_profiles, axis=0)
    sem_profile = np.std(all_profiles, axis=0) / np.sqrt(len(all_profiles))

    # 4. Collect Summary Data and Statistics
    summary_list = []
    print("\n--- Processing Summary Files ---")
    for f in summary_files:
        try:
            df = pd.read_csv(f)
            # Add File_Name column to identify source
            fname = os.path.basename(f).replace("_summary.csv", "")
            df.insert(0, 'File_Name', fname)
            
            # Print file being processed to check for duplicates
            print(f"Processing: {fname} (Rows: {len(df)})")
            
            summary_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if summary_list:
        master_summary_df = pd.concat(summary_list, ignore_index=True)
    else:
        master_summary_df = pd.DataFrame()
        
    print(f"Total rows in aggregated summary: {len(master_summary_df)}")
    
    # Calculate Relative Width if missing (v12 compatibility) - handled automatically if needed

    # 5. Save Result Data (CSV)
    # (1) Integrated Profile Data (Mean, SEM) - v13: Prism compatible (n<=256)
    master_profile_df = pd.DataFrame({'Pos_Norm': x_master, 'Mean_Intensity': mean_profile, 'SEM_Intensity': sem_profile})
    master_profile_df.to_csv(os.path.join(target_dir, "Master_Aggregated_Profile.csv"), index=False, encoding='utf-8-sig')
    
    # (2) Integrated Summary Data for Individual Samples
    master_summary_df.to_csv(os.path.join(target_dir, "Master_Aggregated_Summary.csv"), index=False, encoding='utf-8-sig')

    # 6. Visualization (Master Plot)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # [Graph 1] Average Intensity Profile
    for i in range(len(all_profiles)):
        axes[0].plot(x_master, all_profiles[i], color='gray', alpha=0.1, linewidth=0.5)
    
    axes[0].plot(x_master, mean_profile, color='firebrick', linewidth=3, label='Grand Mean')
    axes[0].fill_between(x_master, mean_profile - sem_profile, mean_profile + sem_profile, color='firebrick', alpha=0.2)
    
    # Calculate Mean Width if available
    mean_width_text = ""
    if 'w_px' in master_summary_df.columns:
        # Physical width (um) for text display
        mean_width_um = np.mean(master_summary_df['w_px']) * 0.606
        sem_width_um = (np.std(master_summary_df['w_px']) * 0.606) / np.sqrt(len(master_summary_df['w_px']))
        mean_width_text = f"Mean Length: {mean_width_um:.1f} ± {sem_width_um:.1f} wm"
        
        # Display Mean Width Text
        axes[0].text(0.05, 0.95, mean_width_text, transform=axes[0].transAxes, 
                    fontsize=12, fontweight='bold', color='blue', 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # --- Visualization on Graph (Blue Dotted Line) ---
        # Draw a line representing the Mean Length on the graph ("graph to graph")
        
        if 'r' in master_summary_df.columns and 'l' in master_summary_df.columns:
            # Calculate average width in normalized coordinates directly from indices
            avg_width_indices = np.mean(master_summary_df['r'] - master_summary_df['l'])
            avg_norm_width = (avg_width_indices / 255.0) * 3.0
            
            # Find height where curve width matches avg_norm_width
            peak_val = np.max(mean_profile)
            baseline = np.min(mean_profile)
            best_h = baseline + (peak_val - baseline) * 0.5 # default midpoint
            found_h = False
            
            # Scan from peak down
            for h in np.linspace(peak_val, baseline, 100):
                above = np.where(mean_profile > h)[0]
                if len(above) > 1:
                    w_idx = above[-1] - above[0]
                    # Convert index width to normalized width
                    curr_norm_w = (w_idx / 255.0) * 3.0
                    
                    if curr_norm_w >= avg_norm_width:
                        best_h = h
                        found_h = True
                        break
            
            if found_h:
                # Find exact positions at this height
                above = np.where(mean_profile > best_h)[0]
                l_pos = x_master[above[0]]
                r_pos = x_master[above[-1]]
                
                # Draw Blue Dotted Line representing the Mean Length
                axes[0].plot([l_pos, r_pos], [best_h, best_h], 
                            color='blue', linestyle=':', linewidth=3, alpha=0.8)
                
                # Add vertical ticks at ends
                tick_h = (peak_val - baseline) * 0.04
                axes[0].plot([l_pos, l_pos], [best_h - tick_h, best_h + tick_h], color='blue', lw=2)
                axes[0].plot([r_pos, r_pos], [best_h - tick_h, best_h + tick_h], color='blue', lw=2)

    axes[0].axvline(-1, color='black', linestyle='--', alpha=0.3); axes[0].axvline(1, color='black', linestyle='--', alpha=0.3)
    axes[0].set_title(f"Averaged Intensity Profile (n={len(all_profiles)})")
    axes[0].set_xlabel("Normalized Position (Organoid Border = ±1.0)"); axes[0].set_ylabel("Intensity (a.u.)")
    axes[0].legend(loc='lower right')

    # [Graph 2] Feature Width Comparison
    if 'w_px' in master_summary_df.columns:
        # Convert pixel to um (0.606)
        widths_um = master_summary_df['w_px'] * 0.606
        axes[1].boxplot(widths_um, patch_artist=True, boxprops=dict(facecolor='lightgray', alpha=0.5))
        axes[1].scatter(np.ones(len(widths_um)), widths_um, color='red', alpha=0.6, zorder=3)
        axes[1].set_title("Distribution of Measured Feature Lengths")
        axes[1].set_ylabel("Length (um)")
        axes[1].set_xticklabels(['Analyzed Features'])

    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "Master_Statistical_Analysis.png"), dpi=300)
    plt.show()

    print(f"\n--- Analysis Complete ---")
    print(f"1. Integrated Profile CSV: Master_Aggregated_Profile.csv")
    print(f"2. Aggregated Summary CSV: Master_Aggregated_Summary.csv")
    print(f"3. Statistical Graph: Master_Statistical_Analysis.png")

if __name__ == "__main__":
    run_master_analysis()