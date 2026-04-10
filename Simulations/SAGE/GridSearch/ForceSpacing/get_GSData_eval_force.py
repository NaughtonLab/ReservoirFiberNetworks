import os
import pickle
import numpy as np
import pandas as pd
from scipy.special import legendre
from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

'''
This script is used to load the simulation data for the Force-Spacing grid search, preprocess the data, 
and evaluate the nonlinearity and memory capacities of the system. The results are saved in a dataframe 
and also in individual npz files for each simulation index. The evaluation can be done using either linear 
regression or ridge regression, and the parameters for the evaluation can be easily changed by uncommenting 
the relevant lines in the code. This script is designed to be run after the simulations have been completed 
and the data has been saved in the specified format.
'''

def load_simulation_data(file_path, file_type, start, num_horizontal_threads, num_vertical_threads, step):
    input_data = []
    output_data = []
    time_data = []
    try:
        if file_type == 'pickle':
            with open(f'{file_path}.{file_type}', 'rb') as f:
                data = pickle.load(f)
                data_loaded = True

            rods_history, force_profile = data
            force_profile = np.array(force_profile)
        elif file_type == 'npz':
            npz_file = f'{file_path}.{file_type}'
            if not os.path.exists(npz_file):
                print('File does not exist', npz_file)
                return [np.nan], [np.nan], [np.nan]
            data = np.load(npz_file, allow_pickle=True)
            data_loaded = True
            rods_history = data['rods_history']
            force_profile = data['force_profile']
            force_profile = force_profile.T
            seed_value = data['seed_value']
        else:
            data_loaded = False
            raise NotImplementedError ("This unit scaling has not been implemented")
        if data_loaded:
            print(f"Data loaded successfully for file {file_path}")

            time = np.array(rods_history[0]["time"])
            stop = len(time)
            t = time[start:stop:step]

            seed_value = 1234
            np.random.seed(seed_value)

            duration = np.max(time)
            sample_time = np.ceil(duration).astype(int)
            x_sample = np.linspace(0, sample_time, sample_time*5 + 1)
            y_sample = np.random.uniform(-1,1, size=sample_time*5+1)
            y_sample[0] = 0.0    
            spline = CubicSpline(x_sample, y_sample)
            ip = spline(time)
            ip = ip[:, np.newaxis]
            ip = ip[start:stop:step]

            op = preprocess_rod_data(rods_history, num_horizontal_threads, num_vertical_threads)
            op = op[start:stop:step]

            input_data.append(ip)
            output_data.append(op)
            time_data.append(t)
    except Exception as e:
        print(f"Error loading file: {e}")
    return input_data, output_data, time_data

def preprocess_rod_data(rods_history, num_horizontal_threads, num_vertical_threads):
    rod_pos = []
    for i in range(num_horizontal_threads + num_vertical_threads):
        rod_pos.append(np.array(rods_history[i]["position"]))

    n_elem = rod_pos[0].shape[2]-1

    vert_connect_idx = np.linspace(0, n_elem+1, num_horizontal_threads+2)
    vert_connect_idx = vert_connect_idx[1:-1]
    hor_connect_idx = np.linspace(0, n_elem+1, num_vertical_threads+2)
    hor_connect_idx = hor_connect_idx[1:-1]

    num_connection_nodes = num_horizontal_threads * num_vertical_threads
    connection_nodes = []
    for i in range(num_horizontal_threads):
        for j in range(num_vertical_threads):
            connection_nodes.append(rod_pos[i][..., int(hor_connect_idx[j])][..., 0:2])

    connection_nodes = np.hstack([connection_nodes[i] for i in range(len(connection_nodes))])

    hor_midpoints = (num_vertical_threads + 1) * num_horizontal_threads
    ver_midpoints = (num_horizontal_threads + 1) * num_vertical_threads
    num_segment_midpoints = hor_midpoints + ver_midpoints
    segment_midpoints = []

    # Getting data of segment midpoints for horizontal threads
    for i in range(num_horizontal_threads):
        start_node_idx = 0
        end_node_idx = n_elem + 1
        for j in range(num_vertical_threads+1):
            if j <= num_vertical_threads-1:
                midpoint_node_idx = (start_node_idx + vert_connect_idx[j])/2
                start_node_idx = vert_connect_idx[j]
            else:
                midpoint_node_idx = (vert_connect_idx[-1] + end_node_idx)/2
            midpoint_node_idx = int(midpoint_node_idx)
            current_midpoint = rod_pos[i][..., midpoint_node_idx][..., 0:2]
            segment_midpoints.append(current_midpoint)
            
    # Getting data of segment midpoints for vertical threads
    for i in range(num_vertical_threads):
        start_node_idx = 0
        end_node_idx = n_elem + 1
        for j in range(num_horizontal_threads+1):
            if j <= num_horizontal_threads-1:
                midpoint_node_idx = (start_node_idx + hor_connect_idx[j])/2
                start_node_idx = hor_connect_idx[j]
            else:
                midpoint_node_idx = (hor_connect_idx[-1] + end_node_idx)/2
            midpoint_node_idx = int(midpoint_node_idx)
            current_midpoint = rod_pos[i+num_horizontal_threads][..., midpoint_node_idx][..., 0:2]
            segment_midpoints.append(current_midpoint)

    segment_midpoints = np.hstack([segment_midpoints[i] for i in range(len(segment_midpoints))])
    num_outputs = num_connection_nodes + num_segment_midpoints
    output = np.hstack([connection_nodes, segment_midpoints])

    output /= np.mean(output, axis=0) # mean is calculated along the rows i.e., the number of columns stay the same
    output /= np.std(output, axis=0)

    scaler = preprocessing.StandardScaler().fit(output)
    op = scaler.transform(output)

    return op
    
def nonlinearity_testing(input, output, leg_max_order, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    x = output
    capacity_train_list = []
    capacity_test_list = []
    R2_train_list = []
    R2_test_list = []
    for n in range(1, leg_max_order+1):
        leg = legendre(n)
        y = leg(input)
        shape_input = input.shape
        train_size = int(shape_input[0] * (1 - test_size))
        y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

        # Training
        clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)
        
        idx = np.where(abs(y_train_pred) > 1)
        y_train_pred[idx] = np.mean(y)
        y_train[idx, 0] = np.mean(y)
        
        y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)

        MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        capacity_train = 1 - MSE/y2_train
        R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

        # Testing
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 1)
        y_test_pred[idx] = np.mean(y)
        y_test[idx, 0] = np.mean(y)

        y2_test = (1/len(y_test)) * np.sum((y_test-np.mean(y_test))**2)

        MSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
        capacity_test = 1 - MSE/y2_test
        R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

        if R2_test < 0:
            R2_test = 0
        if R2_train < 0:
            R2_train = 0
        if capacity_test < 0:
            capacity_test = 0
        if capacity_train < 0:
            capacity_train = 0

        # print("legendre", n, "MSE", MSE_train, MSE_test, "y2", y2, "capacity", capacity_train, capacity_test)

        # plt.figure(figsize=(15, 5))
        # plt.subplot(121)
        # plt.scatter(input[:train_size, :], y_train, label='True')
        # plt.scatter(input[:train_size, :],y_train_pred, label='Predicted')
        # plt.title(f'Training Legendre {n}')
        # plt.legend()
        # plt.grid()

        # plt.subplot(122)
        # plt.scatter(input[train_size:, :], y_test, label='True')
        # plt.scatter(input[train_size:, :], y_test_pred, label='Predicted')
        # plt.title(f'Testing Legendre {n}')
        # plt.legend()
        # plt.grid()

        # plt.show()     

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def memory_testing(input, output, max_timesteps_back, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    capacity_train_list = []
    capacity_test_list = []
    R2_train_list = []
    R2_test_list = []
    for n in range(0, max_timesteps_back+1):
        x = output[n:]
        if n == 0:
            y = input
        else:
            y = input[:-n]

        y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

        # Training
        clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)
        
        idx = np.where(abs(y_train_pred) > 1)
        y_train_pred[idx] = np.mean(y)
        y_train[idx, 0] = np.mean(y)
        
        y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)

        MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        capacity_train = 1 - MSE/y2_train
        R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

        # Testing
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 1)
        y_test_pred[idx] = np.mean(y)
        y_test[idx, 0] = np.mean(y)

        y2_test = (1/len(y_test)) * np.sum((y_test-np.mean(y_test))**2)

        MSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
        capacity_test = 1 - MSE/y2_test
        R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

        if R2_test < 0:
            R2_test = 0
        if R2_train < 0:
            R2_train = 0
        if capacity_test < 0:
            capacity_test = 0
        if capacity_train < 0:
            capacity_train = 0

        # plt.figure(figsize=(15, 5))
        # plt.subplot(121)
        # plt.scatter(y_train, y_train, label='True')
        # plt.plot(y_train_pred, label='Predicted')
        # plt.title(f'Training Memory {n}')
        # plt.legend()
        # plt.grid()

        # plt.subplot(122)
        # plt.plot(y_test, label='True')
        # plt.plot(y_test_pred, label='Predicted')
        # plt.title(f'Testing Memory {n}')
        # plt.legend()
        # plt.grid()

        # plt.show()

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

        # print("memory", n, R2_train, R2_test, capacity_train, capacity_test)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def nonlinearity_memory_matrix(input, output, leg_max_order, max_timesteps_back, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    capacity_train_matrix = np.zeros((leg_max_order, max_timesteps_back+1))
    capacity_test_matrix = np.zeros((leg_max_order, max_timesteps_back+1))
    R2_train_matrix = np.zeros((leg_max_order, max_timesteps_back+1))
    R2_test_matrix = np.zeros((leg_max_order, max_timesteps_back+1))

    for n in range(1, leg_max_order+1):
        leg = legendre(n)
        for t in range(max_timesteps_back+1):
            x = output[t:]
            if t == 0:
                y = leg(input)
            else:
                y = leg(input[:-t])

            y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)
        
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

            # Training
            clf.fit(x_train, y_train)
            y_train_pred = clf.predict(x_train)
            
            idx = np.where(abs(y_train_pred) > 1)
            y_train_pred[idx] = np.mean(y)
            y_train[idx, 0] = np.mean(y)
            
            y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)
    
            MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
            capacity_train = 1 - MSE/y2_train
            R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)
    
            # Testing
            y_test_pred = clf.predict(x_test)
    
            idx = np.where(abs(y_test_pred) > 1)
            y_test_pred[idx] = np.mean(y)
            y_test[idx, 0] = np.mean(y)
    
            y2_test = (1/len(y_test)) * np.sum((y_test-np.mean(y_test))**2)
    
            MSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
            capacity_test = 1 - MSE/y2_test
            R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)
    
            if R2_test < 0:
                R2_test = 0
            if R2_train < 0:
                R2_train = 0
            if capacity_test < 0:
                capacity_test = 0
            if capacity_train < 0:
                capacity_train = 0

            capacity_train_matrix[n-1, t] = capacity_train
            capacity_test_matrix[n-1, t] = capacity_test
            R2_train_matrix[n-1, t] = R2_train
            R2_test_matrix[n-1, t] = R2_test

    return capacity_train_matrix, capacity_test_matrix, R2_train_matrix, R2_test_matrix


if __name__ == "__main__":

    fps = 250
    folder = os.path.dirname(os.path.abspath(__file__))
    print(folder)
    path = f'{folder}/Data/'

    csv_name = 'GSEvaluation_force_100sec_cap'

    grid_1 = np.load(f'{folder}/force_spacing_grid.npz', allow_pickle=True)
    grid_1 = grid_1['grid']

    grid_2 = np.load(f'{folder}/force_spacing_grid_2.npz', allow_pickle=True)
    grid_2 = grid_2['grid']
    
    regressor = "Rid"
    test_size = 0.25
    alpha = 1e-2

    idx_list = [i for i in range(252)]

    if idx_list[0] == 0:
        df = pd.DataFrame(columns = ['num_threads', 'spacing(mm)', 'length(mm)', 'force_mag(N)', 'nonlinearity train', 'memory train', 'nonlinearity test', 'memory test'])
    else:
        df = pd.read_csv(f"{folder}/{csv_name}.csv")

    for idx in idx_list:
        if idx < 112:
            grid = grid_1
            idx_temp = idx
        else:
            grid = grid_2
            idx_temp = idx - 112

        grid_data = grid[idx_temp]
        num_horizontal_threads = 4
        num_vertical_threads = num_horizontal_threads
        point_force_mag = grid_data[0]
        spacing = grid_data[1]
        thread_length = spacing * (num_vertical_threads+1)

        skip = 800
        suffix = f'spacing{spacing:.4e}m_PF{-point_force_mag:.0e}Nspline_fps250_stepskip{skip}'

        sim_name = f'{num_horizontal_threads}by{num_vertical_threads}rods_{suffix}_{idx}'
        sim_ip_data, sim_op_data, sim_time_data = load_simulation_data(file_path = f"{path}{sim_name}",
                                                                        file_type = 'npz',
                                                                        start = 0,
                                                                        num_horizontal_threads = num_horizontal_threads,
                                                                        num_vertical_threads = num_vertical_threads,
                                                                        step = 1)
                                                                        
        ## Evaluation
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
            # max_timesteps_back = 10
            # max_time_back_seconds = np.rint(max_timesteps_back/sample_freq).astype(int)
            mem_capacity_train_list, mem_capacity_test_list, mem_R2_train_list, mem_R2_test_list = memory_testing(input_data, output_data, max_timesteps_back, regressor, test_size, alpha)

            # Nonlinearity-Memory matrix
            capacity_train_matrix, capacity_test_matrix, R2_train_matrix, R2_test_matrix = nonlinearity_memory_matrix(input_data, output_data, leg_max_order, max_timesteps_back, regressor, test_size, alpha)

            onc_train = sum(leg_capacity_train_list)/len(leg_capacity_train_list)
            omc_train = sum(mem_capacity_train_list)/len(mem_capacity_train_list)
            onc_test = sum(leg_capacity_test_list)/len(leg_capacity_test_list)
            omc_test = sum(mem_capacity_test_list)/len(mem_capacity_test_list)

            # print(onc, omc)
            
        else:
            leg_capacity_train_list = np.nan
            mem_capacity_train_list = np.nan
            capacity_train_matrix = np.nan
            leg_capacity_test_list = np.nan
            mem_capacity_test_list = np.nan
            capacity_test_matrix = np.nan
            leg_R2_train_list = np.nan
            mem_R2_train_list = np.nan
            R2_train_matrix = np.nan
            leg_R2_test_list = np.nan
            mem_R2_test_list = np.nan
            R2_test_matrix = np.nan
            onc_train = np.nan
            omc_train = np.nan
            onc_test = np.nan
            omc_test = np.nan

        # Save results in dataframe
        df.at[idx, 'num_threads'] = num_horizontal_threads
        df.at[idx, 'spacing(mm)'] = spacing*1e3
        df.at[idx, 'length(mm)'] = thread_length*1e3
        df.at[idx, 'force_mag(N)'] = point_force_mag
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

    df.to_csv(f"{folder}/{csv_name}.csv", index=False)