import os
import pandas as pd
import numpy as np

def interpolate_labels(file_path):
    print(f"Processing {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Pivot to wide format to access neighbors easily
    # Index: Time, Columns: MultiIndex (Variable, Label)
    df_wide = df.pivot_table(index='Time', columns='Label', values=['X', 'Y'], aggfunc='first')
    
    # Reindex to ensure we have all time steps (handling temporal gaps)
    all_times = np.arange(df['Time'].min(), df['Time'].max() + 1)
    df_wide = df_wide.reindex(all_times)
    
    # Configuration for labels to fix
    # We process them in order. 
    targets_config = [
        {
            'label': 51,
            'x_neighbors': [41, 60], # Vertical neighbors (share X)
            'y_neighbors': [50, 52]  # Horizontal neighbors (share Y)
        },
        {
            'label': 60,
            'x_neighbors': [51, 70], # Vertical neighbors (share X)
            'y_neighbors': [59, 61]  # Horizontal neighbors (share Y)
        }
    ]

    for config in targets_config:
        target_label = config['label']
        x_neighbors = config['x_neighbors']
        y_neighbors = config['y_neighbors']
        
        # Ensure target label exists in columns (if completely missing)
        if ('X', target_label) not in df_wide.columns:
            df_wide[('X', target_label)] = np.nan
            df_wide[('Y', target_label)] = np.nan

        print(f"Interpolating Label {target_label}...")

        # --- Interpolate X for target_label ---
        # We use the X coordinates of vertical neighbors to estimate X
        x_neighbor_data = []
        for n in x_neighbors:
            if ('X', n) in df_wide.columns:
                x_neighbor_data.append(df_wide[('X', n)])
        
        if x_neighbor_data:
            # Calculate mean of available neighbors (ignoring NaNs)
            x_est = pd.concat(x_neighbor_data, axis=1).mean(axis=1)
            # Fill missing X values for target_label with spatial estimate
            df_wide[('X', target_label)] = df_wide[('X', target_label)].fillna(x_est)
            
        # --- Interpolate Y for target_label ---
        # We use the Y coordinates of horizontal neighbors to estimate Y
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
        df_wide[('X', target_label)] = df_wide[('X', target_label)].interpolate(method='linear', limit_direction='both')
        df_wide[('Y', target_label)] = df_wide[('Y', target_label)].interpolate(method='linear', limit_direction='both')

    # Convert back to long format
    df_long = df_wide.stack(level='Label').reset_index()
    df_long['Label'] = df_long['Label'].astype(int)
    df_long = df_long.sort_values(by=['Time', 'Label'])
    df_long = df_long.dropna(subset=['X', 'Y'])

    # Save back to file
    output_path = file_path.replace('.csv', '_fixed.csv')
    df_long.to_csv(output_path, index=False)
    print(f"Saved fixed data to {output_path}")

if __name__ == "__main__":
    # Specific file requested
    target_file = os.path.join("Experiments", "SAGE", "6by6", "Videos", "2_100_45.MP4_frames", "clean_data_rate2.csv")
    
    if os.path.exists(target_file):
        interpolate_labels(target_file)
    else:
        print(f"File not found: {target_file}")
