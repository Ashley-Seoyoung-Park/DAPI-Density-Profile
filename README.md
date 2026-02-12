# DAPI-Density-Profile
This repository contains Python scripts for analyzing necrotic cores in organoid images. The tools allow for individual organoid analysis (boundary detection, intensity profiling, and metric calculation) and integrated statistical analysis of multiple samples.

## Requirements

- Python 3.x
- Tkinter (usually included with Python)
- NumPy (`numpy`)
- Pandas (`pandas`)
- Matplotlib (`matplotlib`)
- Scikit-image (`scikit-image`)
- SciPy (`scipy`)
- Roifile (`roifile`)

Install dependencies via pip:
```bash
pip install numpy pandas matplotlib scikit-image scipy roifile
```

## Scripts

### 1. `necroticcore_v13.py`
**Purpose:** Analyzes individual organoid images to characterize the necrotic core.
- **Inputs:** A parent directory containing organoid images (`*_ch00.tif`).
- **Key Features:**
    - **Automatic Boundary Detection:** Uses Otsu thresholding and morphological operations.
    - **Optimal Slice Selection:** Finds the most symmetric intensity profile through the organoid center.
    - **Feature Detection:** Identifies peaks (bright centers) and dips (dark necrotic cores) in the intensity profile.
    - **Manual Override:** Supports `line.roi` files for manual specification of the analysis line.
    - **Outputs:**
        - `*_analysis.png`: Visualization of the optimal line and intensity profile.
        - `*_profile.csv`: Normalized intensity profile data.
        - `*_summary.csv`: Summary metrics for the selected feature (width, symmetry, etc.).
        - `*_all_features.csv`: Details of all detected peaks and dips.

### 2. `master_analysis.py`
**Purpose:** Aggregates results from `necroticcore_v13.py` across multiple samples.
- **Inputs:** A parent directory containing the output files from the individual analysis.
- **Key Features:**
    - **Data Aggregation:** Combines all `*_profile.csv` and `*_summary.csv` files.
    - **Normalization:** Interpolates profiles to a standard length (n=256) for compatibility with GraphPad Prism.
    - **Statistical Analysis:** Calculates Mean ± SEM for intensity profiles.
    - **Visualization:** Generates a summary plot showing the Grand Mean profile and feature width distribution.
    - **Outputs:**
        - `Master_Aggregated_Profile.csv`: Combined profile data.
        - `Master_Aggregated_Summary.csv`: Combined summary metrics.
        - `Master_Statistical_Analysis.png`: Summary plots.

## Usage

1. **Individual Analysis:**
   - Run `necroticcore_v13.py`.
   - Select the parent folder containing your organoid images.
   - The script will process all `*_ch00.tif` files recursively.

2. **Integrated Analysis:**
   - After processing all images, run `master_analysis.py`.
   - Select the same parent folder (or the one containing the results).
   - The script will aggregate the data and generate the master report.

## Notes
- The scripts handle Unicode paths (e.g., Korean characters).
- Ensure image filenames end with `_ch00.tif` for automatic detection.
- Intensity profiles are normalized spatially: Center=0, Organoid Edges=±1.0.
