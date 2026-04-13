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
        case "Increasing_Density":
            grid_data = grid[idx]
            num_horizontal_threads = int(grid_data[0])
            num_vertical_threads = num_horizontal_threads
            point_force_mag = grid_data[2]
            spacing = grid_data[1]
            thread_length = 500e-3

            skip = 800
            suffix = f'spacing{spacing:.4e}m_TF1e-02N_PF{-point_force_mag:.0e}Nspline_fps250_stepskip{skip}'

            sim_name = f'{num_horizontal_threads}by{num_vertical_threads}rods_{suffix}_{idx}'

            if update_df:
                df.at[idx, 'num_threads'] = num_horizontal_threads

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
    case_name = "Increasing_Density"
    regenerate_ip = True
    update_df = False
    n_splits = 10
    CV = True
    type_CV_nonlin = 'KFold'
    type_CV_mem = 'TimeSeriesSplit'

    leg_max_order = 10
    max_time_back_seconds = 1
    max_timesteps_back = np.rint(fps*max_time_back_seconds).astype(int)

    match case_name:
        case "Increasing_Density":
            folder = os.path.join(os.getcwd(), 'Simulations/SAGE/IncreasingDensity')
            path = os.path.join(folder, 'Data', '')
            csv_name = f'IncreasingDensity_alpha_sweep_{n_splits}CV'

            regressor = "Rid"
            test_size = 0.25
            alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]

            grid = np.load(f'{folder}/increasing_density_list.npz', allow_pickle=True)
            grid = grid['grid']

            idx_list = [i for i in range(6)]

            columns = ['num_threads', 'alpha', 'nonlinearity train', 'nonlinearity test', 'memory train', 'memory test']

    if os.path.exists(f"{csv_name}.csv"):
        df = pd.read_csv(f"{csv_name}.csv")
    else:
        df = pd.DataFrame(columns=columns)
    
    j = 0
    for idx in idx_list:
        record_dict = {}
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
        # data = np.load(f"{path}{idx}_eval_{n_splits}_alphaCV.npz", allow_pickle=True)

        for alpha in alpha_list:
        
            if not np.isnan(input_data).any():
                print(idx)

                input_data = -1 + (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data)) * (1 - (-1))

                print(input_data.shape, output_data.shape)

                # Nonlinearity testing
                leg_capacity_train_list, leg_capacity_test_list, leg_R2_train_list, leg_R2_test_list = nonlinearity_testing(input_data, output_data, leg_max_order, regressor, test_size, alpha, CV, type_CV_nonlin, n_splits)
                
                # Memory testing
                mem_capacity_train_list, mem_capacity_test_list, mem_R2_train_list, mem_R2_test_list = memory_testing(input_data, output_data, max_timesteps_back, regressor, test_size, alpha, CV, type_CV_mem, n_splits)

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
            df.at[j, 'alpha'] = alpha
            df.at[j, 'nonlinearity train'] = onc_train
            df.at[j, 'memory train'] = omc_train
            df.at[j, 'nonlinearity test'] = onc_test
            df.at[j, 'memory test'] = omc_test

            j += 1

            record_dict[alpha] = (leg_capacity_train_list, leg_capacity_test_list, mem_capacity_train_list, mem_capacity_test_list)

        # Please note that the actual files with suffix "_{n_splits}_alphaCV" in the Data folder do not contain input, output, and time data. 
        # They only contain the record_dict with the evaluation results. These evaluation results can be accessed using the following code:
        # data = np.load(f"{path}{idx}_eval_{n_splits}_alphaCV.npz", allow_pickle=True)
        # data = data['npsavez_dict'].item()
        # metrics = data[f'{alpha}']
        np.savez(f"{path}{idx}_eval_{n_splits}_alphaCV.npz", input_data=input_data, 
                output_data=output_data, 
                time_data=time_data, 
                record_dict=record_dict)
        
        print(idx, "eval done.")

    df.to_csv(f"{csv_name}.csv", index=False)
    print(f"Dataframe saved as {csv_name}.csv")