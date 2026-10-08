import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import os
import glob

class LassoLabelerApp:
    def __init__(self, root, initial_path='.'):
        self.root = root
        self.root.title("Interactive Lasso Labeler & Extractor")
        self.root.geometry("1200x800")

        self.df = None
        self.current_file = None
        self.scat = None
        self.selection_scat = None
        self.selected_indices = set()
        self.lasso = None
        
        # --- Layout ---
        left_panel = ttk.Frame(root, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        right_panel = ttk.Frame(root, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # --- Controls ---
        
        # File Loading
        ttk.Label(left_panel, text="1. Load Data", font='Helvetica 10 bold').pack(anchor=tk.W, pady=(0, 5))
        
        # Folder Selection
        ttk.Label(left_panel, text="Parent Folder:").pack(anchor=tk.W)
        self.path_var = tk.StringVar(value=initial_path)
        self.path_entry = ttk.Entry(left_panel, textvariable=self.path_var)
        self.path_entry.pack(fill=tk.X, pady=(0, 2))
        
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_frame, text="Browse...", command=self.browse_folder).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(btn_frame, text="Scan Files", command=self.scan_files).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        
        # File Dropdown
        ttk.Label(left_panel, text="Select File:").pack(anchor=tk.W)
        self.file_var = tk.StringVar()
        self.file_dropdown = ttk.Combobox(left_panel, textvariable=self.file_var, state="readonly")
        self.file_dropdown.pack(fill=tk.X, pady=(0, 5))
        self.file_dropdown.bind("<<ComboboxSelected>>", self.load_file_from_dropdown)
        
        self.file_label = ttk.Label(left_panel, text="No file loaded", wraplength=180, foreground="gray")
        self.file_label.pack(pady=2)
        
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Selection Tools
        ttk.Label(left_panel, text="2. Selection", font='Helvetica 10 bold').pack(anchor=tk.W, pady=(0, 5))
        
        self.add_mode_var = tk.BooleanVar(value=False)
        self.add_chk = ttk.Checkbutton(left_panel, text="Add to Selection (accumulate)", variable=self.add_mode_var)
        self.add_chk.pack(anchor=tk.W, pady=2)
        
        ttk.Button(left_panel, text="Clear Selection", command=self.clear_selection).pack(fill=tk.X, pady=2)
        
        self.selection_info_var = tk.StringVar(value="0 points selected")
        ttk.Label(left_panel, textvariable=self.selection_info_var).pack(pady=2)

        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Labeling
        ttk.Label(left_panel, text="3. Labeling", font='Helvetica 10 bold').pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(left_panel, text="Assign Label to Selection", command=self.label_selection).pack(fill=tk.X, pady=5)
        
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Saving
        ttk.Label(left_panel, text="4. Export", font='Helvetica 10 bold').pack(anchor=tk.W, pady=(0, 5))
        
        self.save_mode_var = tk.StringVar(value="selected")
        ttk.Radiobutton(left_panel, text="Save Selected Points Only", variable=self.save_mode_var, value="selected").pack(anchor=tk.W)
        ttk.Radiobutton(left_panel, text="Save All Data", variable=self.save_mode_var, value="all").pack(anchor=tk.W)
        
        ttk.Button(left_panel, text="Save to CSV...", command=self.save_file).pack(fill=tk.X, pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left_panel, textvariable=self.status_var, wraplength=180, relief=tk.SUNKEN, padding=5).pack(side=tk.BOTTOM, fill=tk.X)

        # --- Plot Area ---
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Initial scan if path exists
        self.scan_files()

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.path_var.set(folder_selected)
            self.scan_files()

    def scan_files(self):
        path = self.path_var.get()
        self.status_var.set("Scanning for clean_data.csv files...")
        self.root.update_idletasks()
        
        if os.path.isdir(path):
            files = []
            # Limit recursion depth to improve performance
            max_depth = 5
            base_depth = path.rstrip(os.sep).count(os.sep)
            
            for root_dir, dirs, filenames in os.walk(path):
                current_depth = root_dir.rstrip(os.sep).count(os.sep)
                if current_depth - base_depth > max_depth:
                    del dirs[:] 
                    continue
                    
                if "clean_data_final.csv" in filenames:
                    files.append(os.path.join(root_dir, "clean_data_final.csv"))
            
            self.file_dropdown['values'] = files
            if files:
                self.file_dropdown.current(0)
                self.load_file_from_dropdown()
                self.status_var.set(f"Found {len(files)} files.")
            else:
                self.file_dropdown.set('')
                self.status_var.set("No clean_data_final.csv files found.")
        else:
            self.status_var.set("Invalid directory path.")

    def load_file_from_dropdown(self, event=None):
        file_path = self.file_dropdown.get()
        if not file_path: return
        self.load_file(file_path)

    def load_file(self, file_path):
        if not file_path: return
        
        try:
            self.df = pd.read_csv(file_path)
            self.current_file = file_path
            self.file_label.config(text=os.path.basename(file_path))
            
            # Ensure X, Y exist
            if 'X' not in self.df.columns or 'Y' not in self.df.columns:
                messagebox.showerror("Error", "CSV must have 'X' and 'Y' columns.")
                return
            
            # Ensure numeric
            self.df['X'] = pd.to_numeric(self.df['X'], errors='coerce')
            self.df['Y'] = pd.to_numeric(self.df['Y'], errors='coerce')
            self.df = self.df.dropna(subset=['X', 'Y']).reset_index(drop=True)
            
            if 'Label' not in self.df.columns:
                self.df['Label'] = 0 # Default label
            else:
                self.df['Label'] = pd.to_numeric(self.df['Label'], errors='coerce').fillna(0)
                
            self.selected_indices = set()
            self.plot_data()
            self.status_var.set(f"Loaded {len(self.df)} points.")
            self.selection_info_var.set("0 points selected")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def plot_data(self):
        self.ax.clear()
        
        if self.df is None or self.df.empty:
            self.canvas.draw()
            return
            
        x = self.df['X']
        y = self.df['Y']
        c = self.df['Label']
        
        # Main scatter plot
        # Using a colormap that handles discrete labels well
        self.scat = self.ax.scatter(x, y, c=c, cmap='tab20', s=20, alpha=0.8, edgecolors='none')
        
        # Overlay for selection (initially empty)
        # We use a separate scatter plot with red edges to highlight selection
        self.selection_scat = self.ax.scatter([], [], facecolors='none', edgecolors='red', s=40, linewidths=1.5)
        
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title(f"Data: {os.path.basename(self.current_file)}")
        self.ax.grid(True)
        
        # Initialize Lasso
        # useblit=True for better performance
        self.lasso = LassoSelector(self.ax, self.on_lasso_select, useblit=True)
        
        self.canvas.draw()

    def on_lasso_select(self, verts):
        if self.df is None: return
        
        path = Path(verts)
        points = np.column_stack((self.df['X'], self.df['Y']))
        
        # Check which points are inside the lasso path
        mask = path.contains_points(points)
        new_indices = set(np.where(mask)[0])
        
        if self.add_mode_var.get():
            self.selected_indices.update(new_indices)
        else:
            self.selected_indices = new_indices
            
        self.update_selection_visuals()
        self.selection_info_var.set(f"{len(self.selected_indices)} points selected")
        self.status_var.set(f"Lasso selected {len(new_indices)} points.")

    def update_selection_visuals(self):
        if not self.selected_indices:
            self.selection_scat.set_offsets(np.empty((0, 2)))
        else:
            indices = list(self.selected_indices)
            selected_points = self.df.iloc[indices][['X', 'Y']].values
            self.selection_scat.set_offsets(selected_points)
        
        self.canvas.draw_idle()

    def clear_selection(self):
        self.selected_indices = set()
        self.update_selection_visuals()
        self.selection_info_var.set("0 points selected")
        self.status_var.set("Selection cleared.")

    def label_selection(self):
        if not self.selected_indices:
            messagebox.showwarning("Warning", "No points selected to label.")
            return
            
        label = simpledialog.askinteger("Input", "Enter Label ID for selected points:", parent=self.root)
        if label is not None:
            self.df.loc[list(self.selected_indices), 'Label'] = label
            
            # Update the main scatter plot colors
            self.scat.set_array(self.df['Label'])
            self.canvas.draw_idle()
            
            self.status_var.set(f"Labeled {len(self.selected_indices)} points as {label}.")

    def save_file(self):
        if self.df is None: return
        
        # Determine what to save
        if self.save_mode_var.get() == "selected":
            if not self.selected_indices:
                messagebox.showwarning("Warning", "No points selected to save.")
                return
            data_to_save = self.df.iloc[list(self.selected_indices)]
            msg_prefix = "Selected points"
        else:
            data_to_save = self.df
            msg_prefix = "All data"
            
        # Ask for filename
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                                 filetypes=[("CSV Files", "*.csv")],
                                                 title="Save CSV As")
        if not save_path: return
        
        try:
            data_to_save.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"Saved {len(data_to_save)} rows to {save_path}")
            self.status_var.set(f"Saved {msg_prefix} to {os.path.basename(save_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LassoLabelerApp(root)
    root.mainloop()
