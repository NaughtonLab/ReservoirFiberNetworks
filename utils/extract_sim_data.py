import os
import pickle
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_simulation_data(file_path, file_type, start, num_horizontal_threads, num_vertical_threads, step, regenerate_ip=False):
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

            if regenerate_ip:
                print("Regenerating input data using cubic spline interpolation with seed value", seed_value)
                seed_value = 1234
                np.random.seed(seed_value)

                duration = np.max(time)
                sample_time = np.ceil(duration).astype(int)
                x_sample = np.linspace(0, sample_time, sample_time*5 + 1)
                y_sample = np.random.uniform(-1,1, size=sample_time*5+1)
                y_sample[0] = 0.0    
                spline = CubicSpline(x_sample, y_sample)
                ip = spline(time)
            else:
                ip = force_profile

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