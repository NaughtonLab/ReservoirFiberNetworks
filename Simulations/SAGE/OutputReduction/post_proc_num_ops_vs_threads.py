import pickle
import numpy as np
from scipy.special import legendre
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import CubicSpline

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
            data = np.load(f'{file_path}.{file_type}', allow_pickle=True)
            data_loaded = True
            rods_history = data['rods_history']
            force_profile = data['force_profile']
            force_profile = force_profile.T
            # seed_value = data['seed_value']
        else:
            data_loaded = False
            raise NotImplementedError ("This unit scaling has not been implemented")
        if data_loaded:
            print(f"Data loaded successfully for file {file_path}")

            time = np.array(rods_history[0]["time"])
            print(f"Total time steps in simulation: {len(time)}")
            print(f"Duration of simulation: {time[-1]} seconds")
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

    # for i in range(1, 3):
    #     for j in range(1, 3):
    #         connection_nodes.append(rod_pos[i][..., int(hor_connect_idx[j])][..., 0:2])

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

    # for i in range(1, 3):
    #     start_node_idx = hor_connect_idx[0]
    #     end_node_idx = n_elem + 1
    #     for j in range(1, 3+1):
    #         if j <= num_vertical_threads-1:
    #             midpoint_node_idx = (start_node_idx + hor_connect_idx[j])/2
    #             start_node_idx = hor_connect_idx[j]
    #         else:
    #             midpoint_node_idx = (hor_connect_idx[-1] + end_node_idx)/2
    #         midpoint_node_idx = int(midpoint_node_idx)
    #         current_midpoint = rod_pos[i][..., midpoint_node_idx][..., 0:2]
    #         segment_midpoints.append(current_midpoint)
            
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

    # for i in range(1, 3):
    #     start_node_idx = vert_connect_idx[0]
    #     end_node_idx = n_elem + 1
    #     for j in range(1, 3+1):
    #         if j <= num_horizontal_threads-1:
    #             midpoint_node_idx = (start_node_idx + vert_connect_idx[j])/2
    #             start_node_idx = vert_connect_idx[j]
    #         else:
    #             midpoint_node_idx = (vert_connect_idx[-1] + end_node_idx)/2
    #         midpoint_node_idx = int(midpoint_node_idx)
    #         current_midpoint = rod_pos[i+num_horizontal_threads][..., midpoint_node_idx][..., 0:2]
    #         segment_midpoints.append(current_midpoint)

    segment_midpoints = np.hstack([segment_midpoints[i] for i in range(len(segment_midpoints))])
    num_outputs = num_connection_nodes + num_segment_midpoints
    op = np.hstack([connection_nodes, segment_midpoints])

    # WHY DO THIS IF WE ARE GOING TO USE STANDARDSCALER LATER??
    # output /= np.mean(output, axis=0) # mean is calculated along the rows i.e., the number of columns stay the same
    # output /= np.std(output, axis=0)

    # scaler = preprocessing.StandardScaler().fit(output)
    # op = scaler.transform(output)

    return op

def nonlinearity_testing(input_data, output, leg_max_order, regressor, test_size, alpha):
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
        y = leg(input_data)
        shape_input = input_data.shape
        train_size = int(shape_input[0] * (1 - test_size))
        y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
        
        # Training
        clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)

        idx = np.where(abs(y_train_pred) > 1)
        y_train_pred[idx] = np.mean(y)
        y_train[idx] = np.mean(y)

        # Testing
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 1)
        y_test_pred[idx] = np.mean(y)
        y_test[idx] = np.mean(y)

        y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)

        MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        capacity_train = 1 - MSE/y2_train
        R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

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

def memory_testing(input, output, max_timesteps_back, regressor, test_size, alpha, fps):
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
        y_train[idx] = np.mean(y)

        # Testing
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 1)
        y_test_pred[idx] = np.mean(y)
        y_test[idx] = np.mean(y)

        y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)

        MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        capacity_train = 1 - MSE/y2_train
        R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

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
            y_train[idx] = np.mean(y)

            # Testing
            y_test_pred = clf.predict(x_test)

            idx = np.where(abs(y_test_pred) > 1)
            y_test_pred[idx] = np.mean(y)
            y_test[idx] = np.mean(y)

            y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)

            MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
            capacity_train = 1 - MSE/y2_train
            R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

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

if __name__ == '__main__':

    start = 0
    step = 1
    fps = 250

    regressor = "Rid"
    test_size = 0.25
    alpha = 1e-3

    for n in [6]:
        file_path = f"./{n}by{n}rods_spacing1.6666e+02m_PF-2e-01Nspline_fps250_stepskip800"

        input_data, output_data, time_data = load_simulation_data(file_path, 'npz', start, n, n, step)
        print(input_data[0].shape, output_data[0].shape, time_data[0].shape)

        # plt.figure(figsize=(20, 5))
        # plt.subplot(121)
        # plt.plot(time_data[0], input_data[0])
        # plt.xlabel('time')
        # plt.ylabel('input')
        # plt.subplot(122)
        # plt.plot(time_data[0], output_data[0])
        # plt.xlabel('time')
        # plt.ylabel('output')
        # plt.savefig(f"{n}.png")

        op = output_data[0] / np.mean(output_data[0], axis=0) # mean is calculated along the rows i.e., the number of columns stay the same
        op /= np.std(op, axis=0)

        # scaler = preprocessing.StandardScaler().fit(output)
        # op = scaler.transform(output)

        # Nonlinearity testing
        leg_max_order = 10
        leg_capacity_train_list, leg_capacity_test_list, leg_R2_train_list, leg_R2_test_list = nonlinearity_testing(input_data[0], op, leg_max_order, regressor, test_size, alpha)
        
        # Memory testing
        max_time_back_seconds = 1
        max_timesteps_back = np.rint(fps*max_time_back_seconds).astype(int)
        mem_capacity_train_list, mem_capacity_test_list, mem_R2_train_list, mem_R2_test_list = memory_testing(input_data[0], op, max_timesteps_back, regressor, test_size, alpha, fps)

        onc_train = sum(leg_R2_train_list)/len(leg_R2_train_list)
        onc_test = sum(leg_R2_test_list)/len(leg_R2_test_list)

        omc_train = sum(mem_R2_train_list)/len(mem_R2_train_list)
        omc_test = sum(mem_R2_test_list)/len(mem_R2_test_list)

        print(onc_train, onc_test, "and", omc_train, omc_test)

        np.savez(f"./op_groups_6x6_100s/{n}by{n}_orig.npz", input_data=input_data[0], output_data=output_data[0], time_data=time_data[0],
                    legendre_R2=(leg_R2_train_list, leg_R2_test_list), legendre_cap=(leg_capacity_train_list, leg_capacity_test_list),
                    memory_R2=(mem_R2_train_list, mem_R2_test_list), memory_cap=(mem_capacity_train_list, mem_capacity_test_list))

    '''Evaluating 4by4 for lesser outputs'''
    # n = 4
    # n_ops = 16
    # file_path = f"./{n}by{n}rods_spacing1.6666e+02m_PF-9e-02Nspline_fps250_stepskip800"

    # input_data, output_data, time_data = load_simulation_data(file_path, 'npz', start, n, n, step)
    # print(input_data[0].shape, output_data[0].shape, time_data[0].shape)

    # # Nonlinearity testing
    # leg_max_order = 10
    # regressor = "Rid"
    # test_size = 0.25
    # alpha = 0.1
    # leg_capacity_train_list, leg_capacity_test_list, leg_R2_train_list, leg_R2_test_list = nonlinearity_testing(input_data[0], output_data[0], leg_max_order, regressor, test_size, alpha)
    
    # # Memory testing
    # max_time_back_seconds = 1
    # max_timesteps_back = np.rint(fps*max_time_back_seconds).astype(int)
    # regressor = "Rid"
    # test_size = 0.25
    # alpha = 0.1
    # mem_capacity_train_list, mem_capacity_test_list, mem_R2_train_list, mem_R2_test_list = memory_testing(input_data[0], output_data[0], max_timesteps_back, regressor, test_size, alpha, fps)

    # np.savez(f"./{n}by{n}_num_ops_vs_threads_{n_ops}.npz", input_data=input_data[0], output_data=output_data[0], time_data=time_data[0],
    #             legendre_R2=(leg_R2_train_list, leg_R2_test_list), legendre_cap=(leg_capacity_train_list, leg_capacity_test_list),
    #             memory_R2=(mem_R2_train_list, mem_R2_test_list), memory_cap=(mem_capacity_train_list, mem_capacity_test_list))


    '''Evaluating 6by6 for lesser outputs'''
    # n = 6
    # n_ops = 56 #16 #
    # file_path = f"./{n}by{n}rods_spacing1.6666e+02m_PF-9e-02Nspline_fps250_stepskip800"

    # input_data, output_data, time_data = load_simulation_data(file_path, 'npz', start, n, n, step)
    # print(input_data[0].shape, output_data[0].shape, time_data[0].shape)

    # # Nonlinearity testing
    # leg_max_order = 10
    # regressor = "Rid"
    # test_size = 0.25
    # alpha = 0.1
    # leg_capacity_train_list, leg_capacity_test_list, leg_R2_train_list, leg_R2_test_list = nonlinearity_testing(input_data[0], output_data[0], leg_max_order, regressor, test_size, alpha)
    
    # # Memory testing
    # max_time_back_seconds = 1
    # max_timesteps_back = np.rint(fps*max_time_back_seconds).astype(int)
    # regressor = "Rid"
    # test_size = 0.25
    # alpha = 0.1
    # mem_capacity_train_list, mem_capacity_test_list, mem_R2_train_list, mem_R2_test_list = memory_testing(input_data[0], output_data[0], max_timesteps_back, regressor, test_size, alpha, fps)

    # np.savez(f"./{n}by{n}_num_ops_vs_threads_{n_ops}.npz", input_data=input_data[0], output_data=output_data[0], time_data=time_data[0],
    #             legendre_R2=(leg_R2_train_list, leg_R2_test_list), legendre_cap=(leg_capacity_train_list, leg_capacity_test_list),
    #             memory_R2=(mem_R2_train_list, mem_R2_test_list), memory_cap=(mem_capacity_train_list, mem_capacity_test_list))
