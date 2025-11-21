import pickle
import numpy as np
from scipy.special import legendre
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
            ip = np.zeros_like(t)[:, np.newaxis]

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

    # WHY DO THIS IF WE ARE GOING TO USE STANDARDSCALER LATER??
    output /= np.mean(output, axis=0) # mean is calculated along the rows i.e., the number of columns stay the same
    output /= np.std(output, axis=0)

    scaler = preprocessing.StandardScaler().fit(output)
    op = scaler.transform(output)

    return op

if __name__ == '__main__':

    file_path = "Simulations/Tests/test4_long_6by6rods_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_PF-2e-01Nvarying_sine_fps250_stepskip800"

    start = 0
    step = 1
    input_data, output_data, time_data = load_simulation_data(file_path, 'npz', start, 6, 6, step)
    print(input_data[0].shape, output_data[0].shape, time_data[0].shape)

    freq_list = [i for i in range(1, 11)]
    hold_time_freq = 5
    hold_time_freq_list = [(f, hold_time_freq*(f-1), hold_time_freq*(f)) for f in freq_list]
    time_scale = 1

    hold_time_freq_list_scaled = [(hold_time_freq_list[i][0]/time_scale,
                                   hold_time_freq_list[i][1]*time_scale,
                                   hold_time_freq_list[i][2]*time_scale) for i in range(len(hold_time_freq_list))]

    hold_time_freq_array = np.array(hold_time_freq_list_scaled)

    for i in range(len(time_data[0])):
        t = time_data[0][i]
        if t > hold_time_freq_array[-1, 2]:
            temp_t = t
            t = hold_time_freq_array[-1, 2] - (t - hold_time_freq_array[-1, 2]) - 1e-10
        else:
            temp_t = t
        idx = np.where((hold_time_freq_array[:, 1] <= t) & (hold_time_freq_array[:, 2] > t))[0][0]
        freq = hold_time_freq_array[idx, 0]
        t = temp_t
        ip = np.sin(t * (2 * np.pi) * freq)
        input_data[0][i, 0] = ip

    plt.plot(time_data[0], input_data[0])
    plt.xlabel('Time (s)')
    plt.ylabel('Input Force (N)')
    plt.title('Input Force vs Time')
    plt.grid()
    plt.show()

    np.savez(f"Simulations/Tests/test4_long.npz", input_data=input_data[0], output_data=output_data[0], time_data=time_data[0])