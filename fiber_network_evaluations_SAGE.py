import os
import pickle
import numpy as np
import pandas as pd
from scipy.special import legendre
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from utils.extract_sim_data import load_simulation_data
from utils.evaluate import nonlinearity_testing, memory_testing, nonlinearity_memory_matrix

'''
This script is used to load the simulation data for the grid search, preprocess the data,
and evaluate the nonlinearity and memory capacities of the system. The results are saved in a dataframe
and also in individual npz files for each simulation index. The evaluation can be done using either
linear regression or ridge regression, and the parameters for the evaluation can be easily changed by uncommenting
the relevant lines in the main function. The get_sim_name_and_update_data_frame function is used to generate the 
simulation name based on the grid data and also update the dataframe with the relevant parameters for each simulation index.
'''

def get_sim_name_and_update_data_frame(grid_data, case_name, idx, update_df):
    match case_name:
        case "GS_Force_Spacing":

            num_horizontal_threads = 4
            num_vertical_threads = num_horizontal_threads
            point_force_mag = grid_data[0]
            spacing = grid_data[1]
            thread_length = spacing * (num_vertical_threads+1)

            skip = 800
            suffix = f'spacing{spacing:.4e}m_PF{-point_force_mag:.0e}Nspline_fps250_stepskip{skip}'

            sim_name = f'{num_horizontal_threads}by{num_vertical_threads}rods_{suffix}_{idx}'
            
            if update_df:
                df.at[idx, 'num_threads'] = num_horizontal_threads
                df.at[idx, 'spacing(mm)'] = spacing*1e3
                df.at[idx, 'length(mm)'] = thread_length*1e3
                df.at[idx, 'force_mag(N)'] = point_force_mag
        case "Force_Sweep":
            num_horizontal_threads = 4
            num_vertical_threads = num_horizontal_threads
            spacing = 70e-3
            point_force_mag = grid_data
            thread_length = spacing * (num_vertical_threads+1)

            skip = 800
            suffix = f'spacing{spacing:.4e}m_PF{-point_force_mag:.0e}Nspline_fps250_stepskip{skip}'

            if update_df:
                df.at[idx, 'num_threads'] = num_horizontal_threads
                df.at[idx, 'spacing(mm)'] = spacing*1e3
                df.at[idx, 'length(mm)'] = thread_length*1e3
                df.at[idx, 'force_mag(N)'] = point_force_mag

        case "GS_Tension_Spacing":
            num_horizontal_threads = 4
            num_vertical_threads = num_horizontal_threads
            tension = grid_data[0]
            spacing = grid_data[1]*1e-3
            point_force_mag = grid_data[2]
            thread_length = spacing * (num_vertical_threads+1)

            skip = 800
            suffix = f'spacing{spacing:.4e}m_PF{-point_force_mag:.0e}Nspline_fps250_stepskip{skip}'

            sim_name = f'{num_horizontal_threads}by{num_vertical_threads}rods_{suffix}_{idx}'

            if update_df:
                df.at[idx, 'num_threads'] = num_horizontal_threads
                df.at[idx, 'spacing(mm)'] = spacing*1e3
                df.at[idx, 'length(mm)'] = thread_length*1e3
                df.at[idx, 'tension(N)'] = tension
                df.at[idx, 'force_mag(N)'] = point_force_mag

        case "GS_Thread_Spacing_NL":
            num_horizontal_threads = int(grid_data[0])
            num_vertical_threads = num_horizontal_threads
            spacing = grid_data[1]
            point_force_mag = grid_data[2]
            thread_length = spacing * (num_vertical_threads+1)

            skip = 800
            suffix = f'spacing{spacing:.4e}m_PF{-point_force_mag:.0e}Nspline_fps250_stepskip{skip}'

            sim_name = f'{num_horizontal_threads}by{num_vertical_threads}rods_{suffix}_{idx}'

            if update_df:
                df.at[idx, 'num_threads'] = num_horizontal_threads
                df.at[idx, 'spacing(mm)'] = spacing*1e3
                df.at[idx, 'length(mm)'] = thread_length*1e3
                df.at[idx, 'force_mag(N)'] = point_force_mag

        case "GS_Thread_Spacing_MC":
            num_horizontal_threads = int(grid_data[0])
            num_vertical_threads = num_horizontal_threads
            spacing = grid_data[1]
            point_force_mag = grid_data[2]
            thread_length = spacing * (num_vertical_threads+1)

            skip = 800
            suffix = f'spacing{spacing:.4e}m_PF{-point_force_mag:.0e}Nspline_fps250_stepskip{skip}'

            sim_name = f'{num_horizontal_threads}by{num_vertical_threads}rods_{suffix}_{idx}'

            if update_df:
                df.at[idx, 'num_threads'] = num_horizontal_threads
                df.at[idx, 'spacing(mm)'] = spacing*1e3
                df.at[idx, 'length(mm)'] = thread_length*1e3
                df.at[idx, 'force_mag(N)'] = point_force_mag

    return sim_name, num_horizontal_threads, num_vertical_threads, df
            

if __name__ == "__main__":
    
    fps = 250
    case_name = "GS_Thread_Spacing_NL"
    regenerate_ip = True
    update_df = False

    match case_name:
        case "GS_Force_Spacing":
            folder = os.path.join(os.getcwd(), 'Simulations/SAGE/GridSearch/ForceSpacing')
            path = os.path.join(folder, 'Data', '')
            csv_name = 'GSEvaluation_force_100sec_cap'

            regressor = "Rid"
            test_size = 0.25
            alpha = 1e-2

            grid_1 = np.load(f'{folder}/force_spacing_grid.npz', allow_pickle=True)
            grid_1 = grid_1['grid']

            grid_2 = np.load(f'{folder}/force_spacing_grid_2.npz', allow_pickle=True)
            grid_2 = grid_2['grid']

            idx_list = [i for i in range(252)]

            columns = ['num_threads', 'spacing(mm)', 'length(mm)', 'force_mag(N)', 'nonlinearity train', 'memory train', 'nonlinearity test', 'memory test']

        case "Force_Sweep":
            folder = os.path.join(os.getcwd(), 'Simulations/SAGE/ForceSweep')
            path = os.path.join(folder, 'Data', '')

            csv_name = 'ForceCliffSweep_evaluation'

            grid = np.load(f'{folder}/forces_sweep.npz', allow_pickle=True)
            grid = grid['sweep']

            idx_list = [i for i in range(9)]

            columns = ['num_threads', 'spacing(mm)', 'length(mm)', 'force_mag(N)', 'nonlinearity', 'memory']
            
        case "GS_Tension_Spacing":
            folder = os.path.join(os.getcwd(), 'Simulations/SAGE/GridSearch/TensionSpacing')
            path = os.path.join(folder, 'Data', '')
            csv_name = 'GSEvaluation_tension_cap'

            regressor = "Rid"
            test_size = 0.25
            alpha = 1e-2

            grid = np.load(f'{folder}/tension_spacing_force_grid_new.npz', allow_pickle=True)
            grid = grid['grid']

            idx_list = [i for i in range(36)]

            columns = ['num_threads', 'spacing(mm)', 'length(mm)', 'tension(N)', 'force_mag(N)', 'nonlinearity train', 'memory train', 'nonlinearity test', 'memory test']

        case "GS_Thread_Spacing_NL":
            folder = os.path.join(os.getcwd(), 'Simulations/SAGE/GridSearch/ThreadSpacing')
            path = os.path.join(folder, 'Data_NL', '')
            csv_name = os.path.join(folder, 'GSEvaluation_NL_cap')

            regressor = "Rid"
            test_size = 0.25
            alpha = 0.01

            grid = np.load(os.path.join(folder, 'thread_spacing_grid_NL.npz'), allow_pickle=True)
            grid = grid['grid']

            idx_list = [i for i in range(0, 84)]

            columns = ['num_threads', 'spacing(mm)', 'length(mm)', 'force_mag(N)', 'nonlinearity train', 'memory train', 'nonlinearity test', 'memory test']

        case "GS_Thread_Spacing_MC":
            folder = os.path.join(os.getcwd(), 'Simulations/SAGE/GridSearch/ThreadSpacing')
            path = os.path.join(folder, 'Data_MC', '')
            csv_name = os.path.join(folder, 'GSEvaluation_MC_cap')
            
            regressor = "Rid"
            test_size = 0.25
            alpha = 1e-2

            grid = np.load(os.path.join(folder, 'thread_spacing_grid_MC.npz'), allow_pickle=True)
            grid = grid['grid']

            idx_list = [i for i in range(0, 84)]

            columns = ['num_threads', 'spacing(mm)', 'length(mm)', 'force_mag(N)', 'nonlinearity train', 'memory train', 'nonlinearity test', 'memory test']

    if os.path.exists(f"{csv_name}.csv"):
        df = pd.read_csv(f"{csv_name}.csv")
    else:
        df = pd.DataFrame(columns=columns)
    

    for idx in idx_list:
        if case_name == "GS_Force_Spacing":
            if idx < 112:
                grid = grid_1
                idx_temp = idx
            else:
                grid = grid_2
                idx_temp = idx - 112
            grid_data = grid[idx_temp]
        else:
            grid_data = grid[idx]

        sim_name, num_horizontal_threads, num_vertical_threads, df = get_sim_name_and_update_data_frame(grid_data, case_name, idx, update_df)
        sim_ip_data, sim_op_data, sim_time_data = load_simulation_data(file_path = f"{path}{sim_name}",
                                                                        file_type = 'npz',
                                                                        start = 0,
                                                                        num_horizontal_threads = num_horizontal_threads,
                                                                        num_vertical_threads = num_vertical_threads,
                                                                        step = 1,
                                                                        regenerate_ip=regenerate_ip)
                                                                        
        ### Evaluation
        input_data = sim_ip_data[0]
        output_data = sim_op_data[0]
        time_data = sim_time_data[0]

        ### Uncomment the below lines to load previously evaluated data that needs to be re-evaluated to avoid reloading the simulation data and redoing the preprocessing steps. This is useful when we want to change the parameters of the evaluation (e.g., regressor, test size, alpha) and want to quickly get the new evaluation results without having to wait for the data loading and preprocessing steps.
        # data = np.load(f"{path}{idx}_eval.npz", allow_pickle=True)
        # input_data = data['input_data']
        # output_data = data['output_data']
        # time_data = data['time_data']
        # leg_capacity_train_list = data['nonlinearity'][0]
        # mem_capacity_train_list = data['memory'][0]
        # leg_capacity_test_list = data['nonlinearity'][1]
        # mem_capacity_test_list = data['memory'][1]
        
        if not np.isnan(input_data).any():
            print(idx)

            input_data = -1 + (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data)) * (1 - (-1))

            print(input_data.shape, output_data.shape)

            # Nonlinearity testing
            leg_max_order = 10
            leg_capacity_train_list, leg_capacity_test_list, leg_R2_train_list, leg_R2_test_list = nonlinearity_testing(input_data, output_data, leg_max_order, regressor, test_size, alpha)
            
            # Memory testing
            max_time_back_seconds = 1
            max_timesteps_back = np.rint(fps*max_time_back_seconds).astype(int)
            mem_capacity_train_list, mem_capacity_test_list, mem_R2_train_list, mem_R2_test_list = memory_testing(input_data, output_data, max_timesteps_back, regressor, test_size, alpha)

            # Nonlinearity-Memory matrix
            capacity_train_matrix, capacity_test_matrix, R2_train_matrix, R2_test_matrix = nonlinearity_memory_matrix(input_data, output_data, leg_max_order, max_timesteps_back, regressor, test_size, alpha)

            onc_train = sum(leg_capacity_train_list)/len(leg_capacity_train_list)
            omc_train = sum(mem_capacity_train_list)/len(mem_capacity_train_list)
            onc_test = sum(leg_capacity_test_list)/len(leg_capacity_test_list)
            omc_test = sum(mem_capacity_test_list)/len(mem_capacity_test_list)

        else:
            leg_capacity_train_list = np.nan
            leg_R2_train_list = np.nan
            mem_capacity_train_list = np.nan
            mem_R2_train_list = np.nan
            capacity_train_matrix = np.nan
            R2_train_matrix = np.nan
            leg_capacity_test_list = np.nan
            leg_R2_test_list = np.nan
            mem_capacity_test_list = np.nan
            mem_R2_test_list = np.nan
            capacity_test_matrix = np.nan
            R2_test_matrix = np.nan
            onc_train = np.nan
            omc_train = np.nan
            onc_test = np.nan
            omc_test = np.nan

        # Save results in dataframe
        df.at[idx, 'nonlinearity train'] = onc_train
        df.at[idx, 'memory train'] = omc_train
        df.at[idx, 'nonlinearity test'] = onc_test
        df.at[idx, 'memory test'] = omc_test

        np.savez(f"{path}{idx}_eval.npz", input_data=input_data, 
                output_data=output_data, 
                time_data=time_data, 
                nonlinearity=[leg_capacity_train_list, leg_capacity_test_list, leg_R2_train_list, leg_R2_test_list], 
                memory=[mem_capacity_train_list, mem_capacity_test_list, mem_R2_train_list, mem_R2_test_list], 
                heatmap=[capacity_train_matrix, capacity_test_matrix, R2_train_matrix, R2_test_matrix])
        
        print(idx, "eval done.")

    df.to_csv(f"{csv_name}.csv", index=False)
    print(f"Dataframe saved as {csv_name}.csv")