import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import glob
import os
import sys
from sklearn.cluster import DBSCAN

class DataCleanerApp:
    def __init__(self, root, initial_path='.'):
        self.root = root
        self.root.title("Fiber Network Data Cleaner")
        self.root.geometry("1200x800")
        
        self.initial_path = initial_path
        self.df = None
        self.current_file = None
        self.scat = None
        self.selection_scat = None
        self.selected_indices = set()
        self.move_target_idx = None
        
        # --- Layout ---
        # Left Frame for Controls
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Right Frame for Plot
        plot_frame = ttk.Frame(root, padding="10")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Controls ---
        
        # Path Selection
        ttk.Label(control_frame, text="Parent Folder:").pack(anchor=tk.W)
        self.path_var = tk.StringVar(value=initial_path)
        self.path_entry = ttk.Entry(control_frame, textvariable=self.path_var)
        self.path_entry.pack(fill=tk.X, pady=(0, 2))
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_frame, text="Browse...", command=self.browse_folder).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(btn_frame, text="Scan Files", command=self.update_file_list).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # File Dropdown
        ttk.Label(control_frame, text="Select File:").pack(anchor=tk.W)
        self.file_var = tk.StringVar()
        self.file_dropdown = ttk.Combobox(control_frame, textvariable=self.file_var, state="readonly")
        self.file_dropdown.pack(fill=tk.X, pady=(0, 5))
        self.file_dropdown.bind("<<ComboboxSelected>>", self.load_file)

        # Video Name Display
        ttk.Label(control_frame, text="Video Name:").pack(anchor=tk.W, pady=(10, 0))
        self.video_name_var = tk.StringVar(value="N/A")
        ttk.Label(control_frame, textvariable=self.video_name_var, font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Mode Selection
        ttk.Label(control_frame, text="Interaction Mode:").pack(anchor=tk.W)
        self.mode_var = tk.StringVar(value="Select")
        ttk.Radiobutton(control_frame, text="Select (Lasso Delete)", variable=self.mode_var, value="Select", command=self.set_mode).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Move (Click & Drag)", variable=self.mode_var, value="Move", command=self.set_mode).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Add (Click to Create)", variable=self.mode_var, value="Add", command=self.set_mode).pack(anchor=tk.W)
        
        ttk.Button(control_frame, text="Delete Selected Points", command=self.delete_selected).pack(fill=tk.X, pady=(5, 10))

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Labeling
        ttk.Label(control_frame, text="Labeling Configuration").pack(anchor=tk.W)
        
        ttk.Label(control_frame, text="X Ranges (comma sep):").pack(anchor=tk.W)
        self.x_ranges_var = tk.StringVar(value="0, 150, 275, 400, 500, 650, 750, 850, 950, 1100")
        ttk.Entry(control_frame, textvariable=self.x_ranges_var).pack(fill=tk.X)
        
        ttk.Label(control_frame, text="Y Ranges (comma sep):").pack(anchor=tk.W)
        self.y_ranges_var = tk.StringVar(value="0, 100, 200, 320, 450, 540, 700, 800, 950, 1070")
        ttk.Entry(control_frame, textvariable=self.y_ranges_var).pack(fill=tk.X)
        
        ttk.Button(control_frame, text="Apply Labels", command=self.apply_labels).pack(fill=tk.X, pady=(5, 10))

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Processing
        ttk.Button(control_frame, text="Auto-Clean (Keep Max Track)", command=self.auto_clean_tracks).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Interpolate/Extrapolate", command=self.interpolate_data).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Save clean_data.csv", command=self.save_data).pack(fill=tk.X, pady=20)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, wraplength=200).pack(side=tk.BOTTOM)

        # --- Plot ---
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initialize selector (will be recreated in plot_data)
        self.rs = None
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initial Load
        self.update_file_list()

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.path_var.set(folder_selected)
            self.update_file_list()

    def update_file_list(self, event=None):
        path = self.path_var.get()
        self.status_var.set("Searching for files...")
        self.root.update_idletasks()
        
        if os.path.isdir(path):
            files = []
            # Limit recursion depth to improve performance on network drives
            max_depth = 10 # Increased depth limit
            base_depth = path.rstrip(os.sep).count(os.sep)
            
            for root_dir, dirs, filenames in os.walk(path):
                current_depth = root_dir.rstrip(os.sep).count(os.sep)
                if current_depth - base_depth > max_depth:
                    del dirs[:] 
                    continue
                    
                if "raw_data.csv" in filenames:
                    full_path = os.path.join(root_dir, "raw_data.csv")
                    # Store full path but display relative path
                    rel_path = os.path.relpath(full_path, path)
                    files.append(rel_path)
            
            self.file_dropdown['values'] = files
            if files:
                self.file_dropdown.current(0)
                self.load_file()
                self.status_var.set(f"Found {len(files)} files.")
            else:
                self.file_dropdown.set('')
                self.status_var.set("No raw_data.csv files found.")
        else:
            self.status_var.set("Invalid directory path.")

    def get_video_name(self):
        if not self.current_file: return "N/A"
        # Assumes structure .../VideoName_frames/raw_data.csv
        dir_name = os.path.basename(os.path.dirname(self.current_file))
        if dir_name.endswith("_frames"):
            return dir_name[:-7] # remove _frames
        return dir_name

    def load_file(self, event=None):
        file_rel_path = self.file_dropdown.get()
        if not file_rel_path: return
        
        # Reconstruct full path
        base_path = self.path_var.get()
        self.current_file = os.path.join(base_path, file_rel_path)
        
        try:
            print(f"Loading {self.current_file}...")
            self.df = pd.read_csv(self.current_file)
            print(f"Columns: {self.df.columns}")
            print(f"Rows before cleaning: {len(self.df)}")
            
            # Ensure numeric types and handle NaNs immediately
            self.df['X'] = pd.to_numeric(self.df['X'], errors='coerce')
            self.df['Y'] = pd.to_numeric(self.df['Y'], errors='coerce')
            
            if 'Label' in self.df.columns:
                self.df['Label'] = pd.to_numeric(self.df['Label'], errors='coerce')
            
            # Drop rows with invalid X or Y
            initial_len = len(self.df)
            self.df = self.df.dropna(subset=['X', 'Y']).reset_index(drop=True)
            print(f"Rows after cleaning: {len(self.df)}")
            
            if len(self.df) == 0:
                self.status_var.set(f"Loaded {os.path.basename(self.current_file)} but NO valid data found!")
                self.ax.clear()
                self.canvas.draw()
                return

            # Print stats for debugging
            print(f"X Range: {self.df['X'].min()} to {self.df['X'].max()}")
            print(f"Y Range: {self.df['Y'].min()} to {self.df['Y'].max()}")

            self.scat = None # Force full reset
            self.plot_data(reset_view=True)
            
            video_name = self.get_video_name()
            self.video_name_var.set(video_name)
            self.status_var.set(f"Loaded {os.path.basename(self.current_file)} ({len(self.df)} points)")
            
            self.selected_indices = set()
            self.move_target_idx = None
        except Exception as e:
            print(f"Error: {e}")
            self.status_var.set(f"Error loading file: {e}")

    def plot_data(self, reset_view=False):
        if self.df is None or self.df.empty:
            self.ax.clear()
            self.canvas.draw()
            return

        # Performance optimization: Downsample for display if too large
        # We keep the full dataframe for operations, but plot a subset for speed if needed
        # However, for selection to work accurately on indices, we must be careful.
        # For now, we plot all points but use a faster method if possible.
        
        x = self.df['X'].values
        y = self.df['Y'].values
        
        # Determine colors
        if 'Label' in self.df.columns:
            # Fill NaNs in Label with -1
            c = self.df['Label'].fillna(-1).values
            cmap = 'tab20'
        else:
            c = 'blue' # Default color
            cmap = None

        # Check if we need to initialize the plot or update it
        if self.scat is None:
            self.ax.clear()
            # Use rasterized=True for better performance with many points
            self.scat = self.ax.scatter(x, y, c=c, cmap=cmap, s=10, picker=5, rasterized=True)
            
            # Overlay for selection
            self.selection_scat = self.ax.scatter([], [], facecolors='none', edgecolors='red', s=40, linewidths=1.5)

            self.ax.set_xlabel('X position (px)')
            self.ax.set_ylabel('Y position (px)')
            self.ax.grid(True)
            
            # Re-initialize selector because ax.clear() removes it
            self.rs = LassoSelector(self.ax, self.on_select, useblit=True)
            self.set_mode() # Ensure correct active state
            reset_view = True
        else:
            # Update existing scatter
            self.scat.set_offsets(np.c_[x, y])
            if cmap is not None:
                self.scat.set_array(c)
            else:
                self.scat.set_color(c)
            
            # Clear selection visuals on data update
            self.selection_scat.set_offsets(np.empty((0, 2)))
            
        # Update Title
        video_name = self.get_video_name()
        self.ax.set_title(f"File: {os.path.basename(self.current_file)} | Video: {video_name}")

        if reset_view:
            # Manually set limits to ensure data is visible
            # ax.relim() is unreliable for collections in some backends
            if len(x) > 0:
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)
                
                # Add padding
                x_pad = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
                y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
                
                self.ax.set_xlim(x_min - x_pad, x_max + x_pad)
                self.ax.set_ylim(y_min - y_pad, y_max + y_pad)
            
            self.canvas.draw()
        else:
            self.canvas.draw_idle()

    def set_mode(self):
        if not self.rs: return
        mode = self.mode_var.get()
        
        # Toggle selector
        if mode == "Select":
            self.rs.set_active(True)
        else:
            self.rs.set_active(False)
            
        # Toggle picker (hit-testing) on scatter plot to improve performance when not moving points
        if self.scat:
            if mode == "Move":
                self.scat.set_picker(5)
            else:
                self.scat.set_picker(None)

    def on_select(self, verts):
        if self.mode_var.get() != 'Select': return
        if self.df is None: return
        
        path = Path(verts)
        points = np.column_stack((self.df['X'], self.df['Y']))
        
        mask = path.contains_points(points)
        self.selected_indices = set(self.df[mask].index)
        self.status_var.set(f"Selected {len(self.selected_indices)} points.")
        self.update_selection_visuals()

    def update_selection_visuals(self):
        if not self.selected_indices:
            self.selection_scat.set_offsets(np.empty((0, 2)))
        else:
            indices = list(self.selected_indices)
            # Use iloc to get points, ensure we handle if indices are out of bounds (though they shouldn't be)
            try:
                selected_points = self.df.iloc[indices][['X', 'Y']].values
                self.selection_scat.set_offsets(selected_points)
            except Exception:
                pass
        
        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax: return

        if self.mode_var.get() == 'Add':
            self.add_point_at(event.xdata, event.ydata)
            return

        if self.mode_var.get() == 'Move':
            if self.move_target_idx is None:
                # Pick
                if self.scat is None: return
                cont, ind = self.scat.contains(event)
                if cont:
                    # Get the index into the dataframe
                    idx = ind['ind'][0]
                    self.move_target_idx = idx
                    self.status_var.set(f"Point {self.move_target_idx} picked. Click to drop.")
            else:
                # Drop
                self.df.at[self.move_target_idx, 'X'] = event.xdata
                self.df.at[self.move_target_idx, 'Y'] = event.ydata
                
                # Fast update of just the plot data
                offsets = self.scat.get_offsets()
                offsets[self.move_target_idx] = [event.xdata, event.ydata]
                self.scat.set_offsets(offsets)
                self.canvas.draw_idle()
                
                self.move_target_idx = None
                self.status_var.set("Point moved.")

    def add_point_at(self, x, y):
        if self.df is None: return
        
        # Ask for Label
        label = simpledialog.askinteger("Input", "Enter Label ID for new cluster:", parent=self.root)
        if label is None: return
        
        # Determine Time (use min time to ensure it exists at start, interpolation will fill forward)
        if 'Time' in self.df.columns and not self.df['Time'].isnull().all():
            t = self.df['Time'].min()
        else:
            t = 0
            
        # Create new row
        new_row = {'X': x, 'Y': y, 'Label': label, 'Time': t}
        
        # Add to dataframe
        new_df = pd.DataFrame([new_row])
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        
        self.status_var.set(f"Added point at ({x:.1f}, {y:.1f}) for Label {label}")
        
        # Full redraw needed to show new point with correct color
        self.scat = None
        self.plot_data(reset_view=False)

    def delete_selected(self):
        if self.selected_indices:
            n = len(self.selected_indices)
            self.df = self.df.drop(list(self.selected_indices)).reset_index(drop=True)
            self.selected_indices = set()
            self.plot_data()
            self.status_var.set(f"Deleted {n} points.")
        else:
            self.status_var.set("No points selected.")

    def apply_labels(self):
        if self.df is None: return
        
        try:
            x_str = self.x_ranges_var.get()
            y_str = self.y_ranges_var.get()
            
            x_ranges = [float(x.strip()) for x in x_str.split(',')]
            y_ranges = [float(y.strip()) for y in y_str.split(',')]
            
            j = 0
            count = 0
            for m in range(len(y_ranges)-1):
                for k in range(len(x_ranges)-1):
                    idx = (self.df['X'] >= x_ranges[k]) & (self.df['X'] <= x_ranges[k+1]) & \
                          (self.df['Y'] >= y_ranges[m]) & (self.df['Y'] <= y_ranges[m+1])
                    
                    if idx.any():
                        self.df.loc[idx, 'Label'] = j
                        count += idx.sum()
                        j += 1
            
            self.plot_data()
            self.status_var.set(f"Applied labels. {j} groups found, {count} points labeled.")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid range format. Use comma-separated numbers.")

    def auto_clean_tracks(self):
        if self.df is None: return
        if 'Label' not in self.df.columns:
            messagebox.showwarning("Warning", "Please apply labels first.")
            return

        # Parameters (could be asked via dialog)
        # eps: max distance between points to be considered neighbors (in space-time)
        # min_samples: min points to form a cluster
        # time_scale: scaling factor for time dimension (to make it comparable to spatial pixels)
        params = simpledialog.askstring("Auto-Clean Parameters", "Enter eps, min_samples, time_scale (comma sep):", initialvalue="30, 5, 5.0", parent=self.root)
        if not params: return
        
        try:
            eps, min_samples, time_scale = map(float, params.split(','))
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters.")
            return

        labels = self.df['Label'].dropna().unique()
        keep_indices = []
        
        # Keep unlabeled data
        unlabeled_indices = self.df[self.df['Label'].isnull()].index.tolist()
        keep_indices.extend(unlabeled_indices)

        count_removed = 0

        for label in labels:
            # Get subset for this label
            mask = self.df['Label'] == label
            subset = self.df[mask]
            
            if len(subset) == 0: continue

            # Prepare features: X, Y, Time*scale
            if 'Time' not in subset.columns:
                keep_indices.extend(subset.index.tolist())
                continue

            # Create feature matrix for DBSCAN
            # We use X, Y, and scaled Time to cluster trajectories
            features = subset[['X', 'Y']].copy()
            features['T_scaled'] = subset['Time'] * time_scale
            
            # Run DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(features)
            
            # Assign cluster labels to a temporary column in the subset (using .loc to avoid warning)
            subset = subset.copy()
            subset['cluster'] = clustering.labels_
            
            unique_clusters = set(clustering.labels_)
            
            # If only noise (-1) found, we might want to keep everything or nothing.
            # Assuming if no structure is found, we keep the data as is (maybe it's just sparse).
            if len(unique_clusters) == 0 or (len(unique_clusters) == 1 and -1 in unique_clusters):
                keep_indices.extend(subset.index.tolist())
                continue
                
            # Find the cluster with the maximum number of unique time frames
            best_cluster = -2
            max_frames = -1
            
            for c in unique_clusters:
                if c == -1: continue # Skip noise
                
                c_data = subset[subset['cluster'] == c]
                n_frames = c_data['Time'].nunique()
                
                if n_frames > max_frames:
                    max_frames = n_frames
                    best_cluster = c
            
            # If we found a valid cluster, keep it. 
            if best_cluster != -2:
                keep_indices.extend(subset[subset['cluster'] == best_cluster].index.tolist())
                count_removed += len(subset) - len(subset[subset['cluster'] == best_cluster])
            else:
                # Fallback: if only noise was found, keep everything
                keep_indices.extend(subset.index.tolist())

        # Update DataFrame
        self.df = self.df.loc[keep_indices].reset_index(drop=True)
        self.plot_data()
        self.status_var.set(f"Auto-cleaned. Removed {count_removed} points.")
        messagebox.showinfo("Auto-Clean", f"Removed {count_removed} points from secondary tracks.")

    def interpolate_data(self):
        if self.df is None: return
        
        try:
            new_dfs = []
            if 'Label' not in self.df.columns:
                self.status_var.set("Error: 'Label' column missing. Apply labels first.")
                return
                
            labels = self.df['Label'].unique()
            if 'Time' not in self.df.columns:
                self.status_var.set("Error: 'Time' column missing.")
                return
                
            full_time = np.arange(self.df['Time'].min(), self.df['Time'].max() + 1)
            
            for label in labels:
                sub = self.df[self.df['Label'] == label].copy()
                sub = sub.drop_duplicates(subset='Time')
                sub = sub.set_index('Time')
                sub = sub.reindex(full_time)
                sub['Label'] = label
                
                sub['X'] = sub['X'].interpolate(method='linear')
                sub['Y'] = sub['Y'].interpolate(method='linear')
                
                sub['X'] = sub['X'].ffill().bfill()
                sub['Y'] = sub['Y'].ffill().bfill()
                
                sub = sub.reset_index().rename(columns={'index': 'Time'})
                new_dfs.append(sub)
                
            self.df = pd.concat(new_dfs, ignore_index=True)
            self.df = self.df.sort_values(by=['Label', 'Time']).reset_index(drop=True)
            self.plot_data()
            self.status_var.set("Interpolated gaps and extrapolated edges.")
        except Exception as e:
            self.status_var.set(f"Interpolation error: {e}")

    def save_data(self):
        if self.df is not None and self.current_file:
            try:
                dir_name = os.path.dirname(self.current_file)
                save_path = os.path.join(dir_name, "clean_data.csv")
                self.df.to_csv(save_path, index=False)
                self.status_var.set(f"Saved to {save_path}")
                messagebox.showinfo("Success", f"Saved to {save_path}")
            except Exception as e:
                self.status_var.set(f"Error saving: {e}")
                messagebox.showerror("Error", f"Could not save: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    # Set initial path to current directory or specific path
    app = DataCleanerApp(root, initial_path='.')
    root.mainloop()
