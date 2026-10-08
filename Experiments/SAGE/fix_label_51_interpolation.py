import os
import pandas as pd
import numpy as np

def interpolate_label(file_path, target_label=51):
    print(f"Processing {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Pivot to wide format to access neighbors easily
    # Index: Time, Columns: MultiIndex (Variable, Label)
    # aggfunc='first' is safe because clean data should have unique (Time, Label) pairs
    df_wide = df.pivot_table(index='Time', columns='Label', values=['X', 'Y'], aggfunc='first')
    
    # Ensure target label exists in columns (if completely missing)
    if ('X', target_label) not in df_wide.columns:
        df_wide[('X', target_label)] = np.nan
        df_wide[('Y', target_label)] = np.nan

    # Reindex to ensure we have all time steps (handling temporal gaps)
    # This creates rows of NaNs for missing time steps
    all_times = np.arange(df['Time'].min(), df['Time'].max() + 1)
    df_wide = df_wide.reindex(all_times)
    
    # Define neighbors based on user input
    # Vertical neighbors (Same X): 41 (below), 60 (above)
    # Horizontal neighbors (Same Y): 50 (left), 52 (right)
    x_neighbors = [41, 60]
    y_neighbors = [50, 52]
    
    # --- Interpolate X for target_label ---
    # We use the X coordinates of vertical neighbors (41, 60) to estimate X for 51
    x_neighbor_data = []
    for n in x_neighbors:
        if ('X', n) in df_wide.columns:
            x_neighbor_data.append(df_wide[('X', n)])
    
    if x_neighbor_data:
        # Calculate mean of available neighbors (ignoring NaNs)
        x_est = pd.concat(x_neighbor_data, axis=1).mean(axis=1)
        # Fill missing X values for target_label with spatial estimate
        # We only fill NaNs, preserving existing data
        df_wide[('X', target_label)] = df_wide[('X', target_label)].fillna(x_est)
        
    # --- Interpolate Y for target_label ---
    # We use the Y coordinates of horizontal neighbors (50, 52) to estimate Y for 51
    y_neighbor_data = []
    for n in y_neighbors:
        if ('Y', n) in df_wide.columns:
            y_neighbor_data.append(df_wide[('Y', n)])
            
    if y_neighbor_data:
        # Calculate mean of available neighbors (ignoring NaNs)
        y_est = pd.concat(y_neighbor_data, axis=1).mean(axis=1)
        # Fill missing Y values for target_label with spatial estimate
        df_wide[('Y', target_label)] = df_wide[('Y', target_label)].fillna(y_est)

    # --- Temporal Interpolation ---
    # Fill any remaining gaps (where neighbors were also missing) using linear interpolation
    # limit_direction='both' ensures extrapolation if needed
    df_wide[('X', target_label)] = df_wide[('X', target_label)].interpolate(method='linear', limit_direction='both')
    df_wide[('Y', target_label)] = df_wide[('Y', target_label)].interpolate(method='linear', limit_direction='both')

    # Convert back to long format
    # stack level 1 (Label) to move it from columns to index. Columns become X and Y.
    df_long = df_wide.stack(level='Label').reset_index()
    
    # Sort by Time and Label
    df_long = df_long.sort_values(by=['Time', 'Label'])
    
    # Remove rows that are still NaN (e.g. gaps in other labels that we created by reindexing)
    # We want to keep the file clean
    df_long = df_long.dropna(subset=['X', 'Y'])

    # Save back to file
    output_path = file_path.replace('.csv', '_fixed.csv')
    df_long.to_csv(output_path, index=False)
    print(f"Saved fixed data to {output_path}")

def process_folder(root_folder):
    print(f"Scanning folder: {root_folder}")
    for root, dirs, files in os.walk(root_folder):
        if "2_2_30.MP4_frames" in root:
            for file in files:
                if file == 'clean_data_rate2.csv':
                    file_path = os.path.join(root, file)
                    interpolate_label(file_path)
        # for file in files:
        #     if file == 'clean_data_rate2.csv':
        #         file_path = os.path.join(root, file)
        #         interpolate_label(file_path)

if __name__ == "__main__":
    # Use absolute path or relative path from workspace root
    folder_to_scan = os.path.join("Experiments", "SAGE", "6by6")
    if os.path.exists(folder_to_scan):
        process_folder(folder_to_scan)
    else:
        print(f"Folder not found: {folder_to_scan}")
