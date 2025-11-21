import pickle
import numpy as np
from scipy.special import legendre
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':
    matplotlib.rc('pdf', fonttype=42)

    spec_label_list = ["2by2", "3by3", "4by4", "6by6", "8by8", "10by10"]
    YM_list = [10, 100]

    figym_leg = plt.figure(figsize=(7.5, 7.5))
    axym_leg = plt.axes()

    figym_mem = plt.figure(figsize=(7.5, 7.5))
    axym_mem = plt.axes()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # ['#6a00a8',  # deep purple
    #         '#a12a8d',  # rich magenta
    #         '#d8576b',  # coral red
    #         '#f9844a',  # warm orange
    #         '#f9c74f',  # sunflower yellow
    #         '#90be6d']  # soft green

    #['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00']
    #['#365c8d', '#277f8e', '#1fa187', '#4ac16d', '#a0da39', '#fde725']
    #['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c']
    #['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725', '#ff9900']

    for YM in YM_list:
        file_path = f'SMASIS_sims/all_post_proc_1sf_YM{YM}/'
        every_x_frame = 1
        sample_freq = np.rint(250/every_x_frame).astype(int)

        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)
        figleg = plt.figure(figsize=(7.5, 7.5))
        axleg = plt.axes()
        figmem = plt.figure(figsize=(7.5, 7.5))
        axmem = plt.axes()

        leg_max_order = 10
        max_time_back_seconds = 1
        max_timesteps_back = np.rint(sample_freq*max_time_back_seconds).astype(int)

        leg_sum_list_ym = []
        mem_sum_list_ym = [] 

        for i in range(len(spec_label_list)):
            spec_label = spec_label_list[i]
            data = np.load(f"{file_path}/{spec_label}_legendre_memory_testing_YM{YM}MPa.npz", allow_pickle=True)
            leg_capacity_test_list = data['leg_capacity_test_list']
            mem_capacity_test_list = data['mem_capacity_test_list']

            leg_sum_list_ym.append(sum(leg_capacity_test_list))
            mem_sum_list_ym.append(sum(mem_capacity_test_list))

            # axleg.plot(np.linspace(1, leg_max_order, leg_max_order), leg_capacity_test_list, '-o', markersize = 10, linewidth = 2, label=spec_label)

            # axmem.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_capacity_test_list, label=spec_label)

        # axleg.set_xlabel('Legendre Polynomial Order')
        # axleg.set_ylabel('Capacity')
        # axleg.set_xticks(np.linspace(1, leg_max_order, leg_max_order))
        # axleg.set_title(f'Variation of Nonlinearity Capacity with Network size for {YM} MPa')
        # axleg.set_ylim(-0.1, 1.1)
        # axleg.legend()
        # # axleg.grid()
        # figleg.savefig(f"SMASIS_sims/SMASIS simulation results/{YM}MPa/legendre_evaluation_network_density_comparison_YM{YM}MPa.pdf", dpi=300)
        plt.close()

        # axmem.set_xlabel('Seconds in the Past')
        # axmem.set_ylabel('Capacity')
        # axmem.set_xticks(np.linspace(0, max_time_back_seconds, 10+1))
        # axmem.set_title(f'Variation of Memory Capacity with Network size for {YM} MPa')
        # axmem.set_ylim(-0.1, 1.1)
        # axmem.legend()
        # # axmem.grid()
        # figmem.savefig(f"SMASIS_sims/SMASIS simulation results/{YM}MPa/memory_evaluation_network_density_comparison_YM{YM}MPa.pdf", dpi=300)
        plt.close()

        axym_leg.plot(spec_label_list, leg_sum_list_ym, '-o', markersize = 10, linewidth = 2, label=f'{YM} MPa')

        axym_mem.plot(spec_label_list, mem_sum_list_ym, '-o', markersize = 10, linewidth = 2, label=f'{YM} MPa')

    axym_leg.set_xlabel('Network Size')
    axym_leg.set_ylabel('Overall Capacity')
    axym_leg.set_title('Variation of Overall Nonlinearity Capacity with Network size')
    axym_leg.legend()
    # axym_leg.grid()
    axym_leg.set_ylim(0, 10)
    figym_leg.savefig(f"SMASIS_sims/SMASIS simulation results/legendre_evaluation_network_density_comparison_YM_all_changedylim.pdf", dpi=300)
    plt.close()

    axym_mem.set_xlabel('Network Size')
    axym_mem.set_ylabel('Overall Capacity')
    axym_mem.set_title('Variation of Overall Memory Capacity with Network size')
    axym_mem.legend()
    # axym_mem.grid()
    axym_mem.set_ylim(0, 250)
    figym_mem.savefig(f"SMASIS_sims/SMASIS simulation results/memory_evaluation_network_density_comparison_YM_all_changedylim.pdf", dpi=300)
    plt.close()


