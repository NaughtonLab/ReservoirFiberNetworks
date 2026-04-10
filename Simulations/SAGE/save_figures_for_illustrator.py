import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FormatStrFormatter

def save_heatmap_figure(heatmap_data, labels, filename, vmin, vmax):
    matplotlib.rc('pdf', fonttype=42)
    x_label = labels['x']
    y_label = labels['y']
    title = labels['title']
    cbar_label = labels['cbar']

    plt.figure(figsize=(7.5, 7.5))
    ax = plt.gca()
    sns.heatmap(heatmap_data, annot=False, cmap='RdYlBu_r', cbar_kws={'label': cbar_label, 'shrink': 0.8}, square=True, vmin=vmin, vmax=vmax)
    ax.set_xlabel(x_label)
    ax.set_xticklabels([f"{val:.3f}" for val in heatmap_data.columns])
    ax.set_ylabel(y_label)
    ax.set_yticklabels([f"{val:.0f}" for val in heatmap_data.index])
    ax.set_title(title)
    cbar = ax.collections[0].colorbar
    # Get the current limits
    vmin, vmax = cbar.mappable.get_clim()
    # Generate ticks at 0.1 intervals
    ticks = np.arange(np.ceil(vmin * 10) / 10, np.floor(vmax * 10) / 10 + 0.05, 0.1)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.savefig(filename, dpi=300)

def save_scatter_plot_figure(x, y, x_label, y_label, title, filename, logx, logy):
    matplotlib.rc('pdf', fonttype=42)
    plt.figure(figsize=(7.5, 7.5))
    plt.scatter(x, y)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, dpi=300)

if __name__ == "__main__":

    one_drive_folder = "C:/D/ABHYAAS/OneDrive - Virginia Tech/naughtonlab - active_projects/2025-JIMSS_paper/pdf_v3/Fig 2"
    heatmaps = True
    scatter = False

    """Force Spacing Grid Search"""

    ## Heatmaps
    data_folder = os.path.join(os.path.dirname(__file__), "GridSearch", "ForceSpacing")
    print("Data folder:", data_folder)
    GS_data = pd.read_csv(os.path.join(data_folder, "GSEvaluation_force_100sec_cap.csv"))

    if heatmaps:

        heatmap_data = GS_data.pivot(index='spacing(mm)', columns='force_mag(N)', values='nonlinearity test')
        labels = {'x': 'Force Magnitude (N)', 'y': 'Spacing (mm)', 'title': 'Force Spacing Grid Search', 'cbar': 'Nonlinearity'}
        filename = os.path.join(one_drive_folder, "ForceSpacing_Nonlinearity.pdf")
        vmin = 0.3 #np.floor(np.min(GS_data['nonlinearity test']) * 10) / 10
        vmax = 0.9 #np.ceil(np.max(GS_data['nonlinearity test']) * 10) / 10
        print("vmin:", vmin, "vmax:", vmax)
        save_heatmap_figure(heatmap_data, labels, filename, vmin=vmin, vmax=vmax)

        print("Saved figure:", filename)

        heatmap_data = GS_data.pivot(index='spacing(mm)', columns='force_mag(N)', values='memory test')
        labels = {'x': 'Force Magnitude (N)', 'y': 'Spacing (mm)', 'title': 'Force Spacing Grid Search', 'cbar': 'Memory'}
        filename = os.path.join(one_drive_folder, "ForceSpacing_Memory.pdf")
        vmin = 0.1 #np.floor(np.min(GS_data['memory test']) * 10) / 10
        vmax = 0.4 #np.ceil(np.max(GS_data['memory test']) * 10) / 10
        print("vmin:", vmin, "vmax:", vmax)
        save_heatmap_figure(heatmap_data, labels, filename, vmin=vmin, vmax=vmax)

        print("Saved figure:", filename)

    if scatter:
        ## Scatter Plots
        x = GS_data['force_mag(N)'] * (GS_data['spacing(mm)']**3)
        y = GS_data['nonlinearity test']
        x_label = r'$F_{ex} \cdot s^3$'
        y_label = 'Nonlinearity'
        title = 'Variation of Nonlinearity with Force and Spacing'
        filename = os.path.join(one_drive_folder, "ForceSpacing_Nonlinearity_Scatter.pdf")
        logx = True
        logy = False
        save_scatter_plot_figure(x, y, x_label, y_label, title, filename, logx, logy)

        print("Saved figure:", filename)

        y = GS_data['memory test']
        y_label = 'Memory'
        title = 'Variation of Memory with Force and Spacing'
        filename = os.path.join(one_drive_folder, "ForceSpacing_Memory_Scatter.pdf")
        logx = True
        logy = False
        save_scatter_plot_figure(x, y, x_label, y_label, title, filename, logx, logy)

        print("Saved figure:", filename)

    """Thread Spacing Grid Search"""
    # data_folder = os.path.join(os.path.dirname(__file__), "GridSearch", "ThreadSpacing")
    # print("Data folder:", data_folder)
    # GS_data = pd.read_csv(os.path.join(data_folder, "GSEvaluation_NL_cap.csv"))
    # GS_data = GS_data.fillna(0)

    # if heatmaps:

    #     ## Set 1
    #     heatmap_data = GS_data.pivot(index='spacing(mm)', columns='num_threads', values='nonlinearity test')
    #     labels = {'x': 'Number of Threads', 'y': 'Spacing (mm)', 'title': 'Thread Spacing Grid Search', 'cbar': 'Nonlinearity'}
    #     filename = os.path.join(one_drive_folder, "ThreadSpacing_Nonlinearity_Set1.pdf")
    #     vmin = np.floor(np.min(GS_data['nonlinearity test']) * 10) / 10
    #     vmax = np.ceil(np.max(GS_data['nonlinearity test']) * 10) / 10
    #     print("vmin:", vmin, "vmax:", vmax)
    #     save_heatmap_figure(heatmap_data, labels, filename, vmin=vmin, vmax=vmax)

    #     print("Saved figure:", filename)

    #     heatmap_data = GS_data.pivot(index='spacing(mm)', columns='num_threads', values='memory test')
    #     labels = {'x': 'Number of Threads', 'y': 'Spacing (mm)', 'title': 'Thread Spacing Grid Search', 'cbar': 'Memory'}
    #     filename = os.path.join(one_drive_folder, "ThreadSpacing_Memory_Set1.pdf")
    #     vmin = np.floor(np.min(GS_data['memory test']) * 10) / 10
    #     vmax = np.ceil(np.max(GS_data['memory test']) * 10) / 10
    #     print("vmin:", vmin, "vmax:", vmax)
    #     save_heatmap_figure(heatmap_data, labels, filename, vmin=vmin, vmax=vmax)

    #     print("Saved figure:", filename)

    #     ## Set 2
    #     heatmap_data = GS_data.pivot(index='spacing(mm)', columns='num_threads', values='nonlinearity test')
    #     labels = {'x': 'Number of Threads', 'y': 'Spacing (mm)', 'title': 'Thread Spacing Grid Search', 'cbar': 'Nonlinearity'}
    #     filename = os.path.join(one_drive_folder, "ThreadSpacing_Nonlinearity_Set2.pdf")
    #     vmin = np.floor(np.min(GS_data['nonlinearity test'][GS_data['nonlinearity test'] > 0]) * 10) / 10
    #     vmax = np.ceil(np.max(GS_data['nonlinearity test']) * 10) / 10
    #     print("vmin:", vmin, "vmax:", vmax)
    #     save_heatmap_figure(heatmap_data, labels, filename, vmin=vmin, vmax=vmax)

    #     print("Saved figure:", filename)

    #     heatmap_data = GS_data.pivot(index='spacing(mm)', columns='num_threads', values='memory test')
    #     labels = {'x': 'Number of Threads', 'y': 'Spacing (mm)', 'title': 'Thread Spacing Grid Search', 'cbar': 'Memory'}
    #     filename = os.path.join(one_drive_folder, "ThreadSpacing_Memory_Set2.pdf")
    #     vmin = np.floor(np.min(GS_data['memory test'][GS_data['memory test'] > 0]) * 10) / 10
    #     vmax = np.ceil(np.max(GS_data['memory test']) * 10) / 10
    #     print("vmin:", vmin, "vmax:", vmax)
    #     save_heatmap_figure(heatmap_data, labels, filename, vmin=vmin, vmax=vmax)

    #     print("Saved figure:", filename)

    # """Tension Spacing Grid Search"""
    # data_folder = os.path.join(os.path.dirname(__file__), "GridSearch", "TensionSpacing")
    # print("Data folder:", data_folder)
    # GS_data = pd.read_csv(os.path.join(data_folder, "GSEvaluation_tension_cap.csv"))

    # if heatmaps:

    #     heatmap_data = GS_data.pivot(index='spacing(mm)', columns='tension(N)', values='nonlinearity test')
    #     labels = {'x': 'Tension (N)', 'y': 'Spacing (mm)', 'title': 'Tension Spacing Grid Search', 'cbar': 'Nonlinearity'}
    #     filename = os.path.join(one_drive_folder, "TensionSpacing_Nonlinearity.pdf")
    #     vmin = np.floor(np.min(GS_data['nonlinearity test']) * 10) / 10
    #     vmax = np.ceil(np.max(GS_data['nonlinearity test']) * 10) / 10
    #     print("vmin:", vmin, "vmax:", vmax)
    #     save_heatmap_figure(heatmap_data, labels, filename, vmin=vmin, vmax=vmax)

    #     print("Saved figure:", filename)

    #     heatmap_data = GS_data.pivot(index='spacing(mm)', columns='tension(N)', values='memory test')
    #     labels = {'x': 'Tension (N)', 'y': 'Spacing (mm)', 'title': 'Tension Spacing Grid Search', 'cbar': 'Memory'}
    #     filename = os.path.join(one_drive_folder, "TensionSpacing_Memory.pdf")
    #     vmin = np.floor(np.min(GS_data['memory test']) * 10) / 10
    #     vmax = np.ceil(np.max(GS_data['memory test']) * 10) / 10
    #     print("vmin:", vmin, "vmax:", vmax)
    #     save_heatmap_figure(heatmap_data, labels, filename, vmin=vmin, vmax=vmax)

    #     print("Saved figure:", filename)