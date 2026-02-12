
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from skimage import io, color, filters, measure, morphology
from skimage.measure import profile_line
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.stats import skew
import roifile  # For loading ImageJ ROI files

# [Settings] Fixed Parameters
PIXEL_TO_UM = 0.606  # 0.606 um/px
MAX_PROFILE_POINTS = 256  # Prism compatibility

def calculate_symmetry_score(profile):
    """Calculate symmetry score of the profile (Lower is more symmetric)"""
    mid = len(profile) // 2
    left = profile[:mid]
    right = profile[mid:][::-1]  # Reverse
    min_len = min(len(left), len(right))
    if min_len == 0: return float('inf')
    
    # Combine standard deviation of difference and skewness
    diff = np.abs(left[:min_len] - right[:min_len])
    symmetry = np.std(diff) + abs(skew(profile))
    return symmetry

def get_organoid_centroid(binary):
    """Find the centroid of the organoid"""
    label_img = measure.label(binary)
    props = measure.regionprops(label_img)
    if not props: return None
    largest = max(props, key=lambda x: x.area)
    return largest.centroid  # (y, x)

def process_single_organoid(file_path):
    try:
        # 1. Load Image and Preprocess
        image = io.imread(file_path)
        gray = color.rgb2gray(image) if image.ndim == 3 else image
        gray_smoothed = filters.gaussian(gray, sigma=1.2)

        # 2. Organoid Boundary Detection (Otsu only, no convex hull)
        thresh = filters.threshold_otsu(gray_smoothed)
        binary = gray_smoothed > thresh
        binary = morphology.binary_closing(binary, morphology.disk(5))
        binary = morphology.remove_small_objects(binary, min_size=1000)
        # Convex hull removed to preserve actual organoid shape

        # 3. Calculate Centroid and Size
        centroid = get_organoid_centroid(binary)
        if centroid is None: return
        cy, cx = centroid
        
        label_img = measure.label(binary)
        props = measure.regionprops(label_img)
        largest = max(props, key=lambda x: x.area)
        bbox = largest.bbox  # (min_row, min_col, max_row, max_col)
        org_height = bbox[2] - bbox[0]
        org_width = bbox[3] - bbox[1]
        max_dimension = max(org_height, org_width)

        # 4. Check for manual line.roi first
        dir_name = os.path.dirname(file_path)
        roi_files = [f for f in os.listdir(dir_name) if f.lower() in ['line.roi', 'line.zip']]
        
        manual_line_used = False
        if roi_files:
            try:
                roi_path = os.path.join(dir_name, roi_files[0])
                roi_data = roifile.ImagejRoi.fromfile(roi_path)
                if isinstance(roi_data, list):
                    roi_data = roi_data[0]
                
                # Extract line coordinates from ROI
                coords = roi_data.coordinates()
                if len(coords) >= 2:
                    # Use first and last points of the ROI as line endpoints
                    start = (coords[0, 1], coords[0, 0])  # (y, x) format
                    end = (coords[-1, 1], coords[-1, 0])
                    
                    # Calculate angle for visualization
                    dy = end[0] - start[0]
                    dx = end[1] - start[1]
                    best_angle = np.rad2deg(np.arctan2(dy, dx)) % 180
                    
                    # Get profile along manual line
                    best_profile = profile_line(gray, start, end, linewidth=3, mode='constant')
                    best_coords = (start, end)
                    best_symmetry = calculate_symmetry_score(best_profile)
                    best_y = (start[0] + end[0]) / 2
                    
                    manual_line_used = True
                    print(f"  ✓ Using manual line.roi: {roi_files[0]}")
            except Exception as e:
                print(f"  ✗ Failed to load line.roi: {e}, falling back to automatic detection")
                manual_line_used = False
        
        # 4b. If no manual ROI, use automatic symmetry-based search
        if not manual_line_used:
            # Symmetry-based optimal slice + angle search
            # Search location within 30% range, angle 0-180 degrees
            search_range = int(org_height * 0.30)
            y_candidates = np.linspace(max(bbox[0], cy - search_range), 
                                       min(bbox[2], cy + search_range), 15)  # 15 positions
            angle_candidates = np.linspace(0, 180, 19)  # 10 degree step (0, 10, 20, ..., 180)
            
            best_y = cy
            best_angle = 0
            best_symmetry = float('inf')
            best_profile = None
            best_coords = None
            
            img_height, img_width = gray.shape
            
            for y in y_candidates:
                for angle in angle_candidates:
                    # Calculate line endpoints based on angle
                    rad = np.deg2rad(angle)
                    # Line length adapted to organoid size
                    line_length = max_dimension * 1.5
                    dx = line_length * np.cos(rad)
                    dy = line_length * np.sin(rad)
                    
                    # Line Start/End points (based on y, centered on x in both directions)
                    start = (y - dy/2, cx - dx/2)
                    end = (y + dy/2, cx + dx/2)
                    
                    try:
                        test_profile = profile_line(gray, start, end, linewidth=3, mode='constant')
                        if len(test_profile) < 50: continue  # Exclude too short profiles
                        
                        # Check if profile has enough valid data (not too many zeros)
                        if np.sum(test_profile > 0) < len(test_profile) * 0.3: continue
                        
                        symmetry = calculate_symmetry_score(test_profile)
                        
                        # Priority-based Penalty System
                        # Priority 1: Vertical/Horizontal & Symmetric
                        # Priority 2: Vertical/Horizontal & Center ±25-30%
                        # Priority 3: Tilted & Symmetric
                        # Priority 4: Tilted & Center ±25-30%
                        
                        # Angle Check: Vertical/Horizontal (±15 degrees allowed)
                        angle_normalized = min(abs(angle), abs(180 - angle))  # 0-90 range
                        is_cardinal = (angle_normalized < 15) or (abs(angle_normalized - 90) < 15)
                        
                        # Position Check: Within Center ±25%
                        distance_from_center = abs(y - cy) / search_range  # 0-1 normalized
                        in_middle_range = distance_from_center < 0.25  # Within 25%
                        
                        # Assign Penalty based on Priority
                        if is_cardinal and in_middle_range:
                            # Priority 1 & 2: Cardinal + Centered -> Judge by symmetry only
                            priority_penalty = 0.0
                        elif is_cardinal:
                            # Cardinal but off-center -> Small penalty
                            priority_penalty = 0.3 + distance_from_center * 0.5
                        elif in_middle_range:
                            # Priority 3: Tilted but Centered
                            priority_penalty = 1.0
                        else:
                            # Priority 4: Tilted + Off-center
                            priority_penalty = 1.5 + distance_from_center * 0.5
                        
                        # Final Score = Symmetry + Priority Penalty
                        final_score = symmetry + priority_penalty
                        
                        if final_score < best_symmetry:
                            best_symmetry = final_score
                            best_y = y
                            best_angle = angle
                            best_profile = test_profile
                            best_coords = (start, end)
                    except Exception as e:
                        continue
        
        if best_profile is None:
            print(f"Failed to find valid profile for {file_path}")
            return
        
        # 5. Generate ROI lines around optimal line (Parallel lines)
        n_lines = 20
        roi_width = org_height * 0.10  # 10% width around optimal line
        rad = np.deg2rad(best_angle)
        
        # Vector perpendicular to line
        perp_dx = -np.sin(rad) * roi_width / 2
        perp_dy = np.cos(rad) * roi_width / 2
        
        profiles = []
        scan_lines_coords = []
        
        for offset in np.linspace(-1, 1, n_lines):
            # Parallel Shift
            start_shifted = (best_coords[0][0] + offset * perp_dy, 
                           best_coords[0][1] + offset * perp_dx)
            end_shifted = (best_coords[1][0] + offset * perp_dy,
                         best_coords[1][1] + offset * perp_dx)
            
            try:
                profile = profile_line(gray, start_shifted, end_shifted, linewidth=3, mode='constant')
                profiles.append(profile)
                scan_lines_coords.append((start_shifted, end_shifted))
            except:
                continue
        
        if not profiles: return
        
        # 6. Calculate Average Profile
        min_len = min(len(p) for p in profiles)
        avg_raw = np.mean([p[:min_len] for p in profiles], axis=0)
        
        # 7. Downsampling for Prism Compatibility
        original_len = min_len  # Store original pixel length
        downsample_factor = 1.0
        
        if min_len > MAX_PROFILE_POINTS:
            downsample_factor = min_len / MAX_PROFILE_POINTS
            indices = np.round(np.linspace(0, min_len - 1, MAX_PROFILE_POINTS)).astype(int)
            avg_raw = avg_raw[indices]
            min_len = len(avg_raw)
            
        # 7. Detect actual organoid boundaries and normalize correctly
        # Find where signal rises above and falls below background (10% of max intensity)
        threshold = np.max(avg_raw) * 0.10
        above_threshold = avg_raw > threshold
        
        if np.any(above_threshold):
            # Find first and last points above threshold (organoid edges)
            organoid_indices = np.where(above_threshold)[0]
            organoid_start = organoid_indices[0]
            organoid_end = organoid_indices[-1]
            organoid_center = (organoid_start + organoid_end) / 2
            organoid_radius = (organoid_end - organoid_start) / 2
            
            # Normalize: organoid edges map to -1 and +1, extend beyond for context
            x_norm = np.zeros(min_len)
            for i in range(min_len):
                # Distance from center in organoid-radius units
                x_norm[i] = (i - organoid_center) / organoid_radius
        else:
            # Fallback if no signal detected
            print(f"Warning: Could not detect organoid boundaries for {file_path}, using default normalization")
            x_norm = np.linspace(-1.5, 1.5, min_len)

        # 8. Peak and Dip Analysis - Moderate smoothing to preserve feature detection
        win = int(min_len / 5)  # Moderate smoothing (1/5 of profile)
        if win % 2 == 0: win += 1
        if win < 5: win = 5  # Ensure minimum window size
        avg_smooth = savgol_filter(avg_raw, window_length=win, polyorder=2, mode='mirror')
        avg_smooth = np.clip(avg_smooth, 0, None)
        inverted = np.max(avg_smooth) - avg_smooth

        # Adjusted parameters to detect both plateaus and peaks without over-smoothing
        min_distance = min_len // 6   # Minimum distance ~17% of profile
        min_width = min_len // 4      # Minimum width 25% of profile
        
        peaks, _ = find_peaks(avg_smooth, prominence=0.01, distance=min_distance, width=min_width)
        # For peaks: use rel_height=0.35 to measure at mid-height for better plateau capture
        p_res = peak_widths(avg_smooth, peaks, rel_height=0.35)
        
        dips, _ = find_peaks(inverted, prominence=0.01, distance=min_distance, width=min_width)
        # For dips: use rel_height=0.55 to measure in the middle of the dip
        d_res = peak_widths(inverted, dips, rel_height=0.55)

        all_f = []
        for i in range(len(peaks)):
            # Downsampling correction: Convert measurement to actual pixel units
            width_px = p_res[0][i] * downsample_factor
            # p_res[1][i] is measurement height
            all_f.append({'type': 'Peak', 'idx': peaks[i], 'w_px': width_px, 'h': p_res[1][i], 'l': p_res[2][i], 'r': p_res[3][i]})
        for i in range(len(dips)):
            # Downsampling correction
            width_px = d_res[0][i] * downsample_factor
            # d_res[1][i] is measurement height on INVERTED signal
            # Convert back to ORIGINAL signal height
            height_original = np.max(avg_smooth) - d_res[1][i]
            all_f.append({'type': 'Dip', 'idx': dips[i], 'w_px': width_px, 'h': height_original, 'l': d_res[2][i], 'r': d_res[3][i]})

        if not all_f: return
        
        # Filter 1: Only keep features within organoid boundary (-1 to +1 normalized position)
        valid_features = []
        for f in all_f:
            feature_pos = x_norm[f['idx']]
            if -1.0 <= feature_pos <= 1.0:
                valid_features.append(f)
        
        if not valid_features:
            print(f"No features found within organoid boundary for {file_path}")
            return
        
        # Filter 2: Only select centered features (Symmetry Guaranteed) - tolerance tightened
        center_idx = min_len / 2
        center_tolerance = min_len * 0.15  # ±15% range
        centered_features = [f for f in valid_features if abs(f['idx'] - center_idx) < center_tolerance]
        
        if not centered_features:
            print(f"No centered features found for {file_path}, using all valid features within boundary")
            centered_features = valid_features
        
        # Feature selection: prioritize the WIDEST feature regardless of type
        best_f = max(centered_features, key=lambda x: x['w_px'])

        # 9. Save Data
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_name = os.path.dirname(file_path)
        pd.DataFrame(all_f).to_csv(os.path.join(dir_name, f"{base_name}_all_features.csv"), index=False)
        pd.DataFrame([best_f]).to_csv(os.path.join(dir_name, f"{base_name}_summary.csv"), index=False)
        pd.DataFrame({"Pos_Norm": x_norm, "Intensity": avg_smooth}).to_csv(os.path.join(dir_name, f"{base_name}_profile.csv"), index=False)

        # 10. Visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # (Left) Image Visualization
        axes[0].imshow(gray, cmap='gray')
        
        # Show scan lines
        for start, end in scan_lines_coords:
            axes[0].plot([start[1], end[1]], [start[0], end[0]], 
                        color='red', alpha=0.5, linewidth=0.8)
        
        # Highlight Main Optimal Line
        color_code = 'red' if best_f['type'] == 'Peak' else 'cyan'
        axes[0].plot([best_coords[0][1], best_coords[1][1]], 
                    [best_coords[0][0], best_coords[1][0]], 
                    color=color_code, linewidth=4, label='Optimal line')
        
        folder_name = os.path.basename(dir_name)
        axes[0].set_title(f"[{folder_name}] {best_f['type']} | Angle={best_angle:.1f}° | Sym={best_symmetry:.3f}", fontsize=12)
        axes[0].legend()

        # (Right) Graph Visualization
        axes[1].plot(x_norm, avg_raw, color='lightgray', alpha=0.3, label='Raw')
        axes[1].plot(x_norm, avg_smooth, color='black', linewidth=2, label='Smoothed')
        
        # Mark Measurement Width
        l_n, r_n = x_norm[int(best_f['l'])], x_norm[int(best_f['r'])]
        axes[1].hlines(best_f['h'], l_n, r_n, color=color_code, linestyle='--', linewidth=2)
        
        axes[1].set_ylim(0, np.max(avg_smooth) * 1.5)
        axes[1].set_xlim(-1.5, 1.5)
        axes[1].set_title(f"Intensity Profile (n={len(x_norm)} points)", fontsize=14)
        axes[1].set_xlabel("Normalized Position (Center=0)")
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(dir_name, f"{base_name}_analysis.png"), dpi=150)
        plt.close(fig)
        print(f"✓ {base_name} | Angle={best_angle:.1f}° | Symmetry={best_symmetry:.3f} | Width={best_f['w_px']*PIXEL_TO_UM:.1f}μm | Points={len(x_norm)}")

    except Exception as e:
        print(f"✗ Error in {file_path}: {e}")
        import traceback
        traceback.print_exc()

def run_batch():
    root = tk.Tk(); root.withdraw()
    parent = filedialog.askdirectory(title="Select the parent folder to analyze")
    if not parent: return
    files = glob.glob(os.path.join(parent, "**", "*_ch00.tif"), recursive=True)
    print(f"Found {len(files)} images to process\n")
    for f in files: process_single_organoid(f)
    print("\n✓ [Analysis and Visualization Complete]")

if __name__ == "__main__":
    run_batch()
