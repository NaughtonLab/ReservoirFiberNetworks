import pickle
import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_simulation_data(file_path, n_files, start, num_horizontal_threads, num_vertical_threads, step):
    input_data = []
    output_data = []
    time_data = []
    for n in range(n_files):
        try:
            with open(f'{file_path}_{n}.pickle', 'rb') as f:
                data = pickle.load(f)
                rods_history, force_profile = data
                force_profile = np.array(force_profile)
                stop = len(force_profile)
                ip = force_profile[start:stop:step]

                time = np.array(rods_history[0]["time"])
                t = time[start:stop:step]

                op = preprocess_rod_data(rods_history, num_horizontal_threads, num_vertical_threads)
                op = op[start:stop:step]

                input_data.append(ip)
                output_data.append(op)
                time_data.append(t)
        except Exception as e:
            print(f"Error loading file {file_path}_{n}: {e}")
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
        for j in range(num_horizontal_threads):
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
    for n in range(leg_max_order+1):
        leg = legendre(n)
        y = leg(input)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
        clf.fit(x_train, y_train)

        # Training capacity
        y_train_pred = clf.predict(x_train)

        MSE = (1/len(y_train)) * np.sum((y_train_pred - y_train)**2)
        z2 = (1/len(y_train)) * np.sum((y_train)**2)
        capacity_train = 1 - MSE/z2

        # Testing capacity
        y_test_pred = clf.predict(x_test)

        MSE = (1/len(y_test)) * np.sum((y_test_pred - y_test)**2)
        z2 = (1/len(y_test)) * np.sum((y_test)**2)
        capacity_test = 1 - MSE/z2

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)

    return capacity_train_list, capacity_test_list

def memory_testing(input, output, max_time_back, regressor, test_size, alpha):
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
    for n in range(max_time_back+1):
        x = output[n:]
        if n == 0:
            y = input
        else:
            y = input[:-n]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
        clf.fit(x_train, y_train)

        # Training capacity
        y_train_pred = clf.predict(x_train)

        MSE = (1/len(y_train)) * np.sum((y_train_pred - y_train)**2)
        z2 = (1/len(y_train)) * np.sum((y_train)**2)
        capacity_train = 1 - MSE/z2

        # Testing capacity
        y_test_pred = clf.predict(x_test)

        MSE = (1/len(y_test)) * np.sum((y_test_pred - y_test)**2)
        z2 = (1/len(y_test)) * np.sum((y_test)**2)
        capacity_test = 1 - MSE/z2

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)

    return capacity_train_list, capacity_test_list

if __name__ == '__main__':
    scaling_type = "mm_g_s"
    params = {
        'num_horizontal_threads': 2,
        'num_vertical_threads': 2,
        'network_origin': np.zeros((3,)), # network_origin is the center of the network

        'thread_length': 500e-3, # 1 m --> 1e3 mm
        'thread_diameter': 2e-3, # 1 m --> 1e3 mm
        'dx': 20e-3, # 1 m --> 1e3 mm

        'youngs_modulus': 100e6, # 1 Pa = kg /m/s2 --> 1 g/mm/s2 --> 1e-3 mg/mm/ms2
        'density': 1e3, # 1 kg / mm3 --> 1e-6 g/mm3 --> 1e-3 mg/mm3

        'tension_force': 1e-2, # 1 N = kg m/s2 --> 1e6 g mm/s2 --> 1e3 mg mm /ms2  
        'point_force_mag': -2e-2, # 1 N = kg m/s2 --> 1e6 g mm/s2 --> 1e3 mg mm /ms2 
        'SPREAD_PF': True, # whether the force should be a gaussian spread across 5 nodes or just applied at a single point
        'TYPE_PF': "spline", # type of force to be applied 
        'sample_freq_pf': 5, # Sampling frequency for random point force

        'damping_constant': 10, 
        'filter_order': 6,

        'k': 1e9, # translational stiffness of connection
        'kt': 1e9, # rotational stiffness of connection
        'nu': 0.0, # translational damping of connection

        'duration': 15, # 1 s --> 1e3 ms
        'sim_dt': 5e-6, # simulation timestep

        'rendering_fps': 250,
        
        'STOP_AT_NAN': True,
        'CALLBACK': True,
        'VIDEO': True,

        'scaling_type': scaling_type,
        'loc': 'SMASIS_sims/2by2/'
    }

    step_skip = np.rint(1.0 / (params['sim_dt'] * params['rendering_fps'])).astype(int)

    suffix = f"{params['duration']:.0f}sec_L{params['thread_length']:.2e}m_R{params['thread_diameter']/2:.2e}m_dx{params['dx']*1e3:.0f}mm_YM{params['youngs_modulus']:.2e}Pa_Density{params['density']:.2e}kgmm-3_Damping{params['damping_constant']:.0f}_TF{params['tension_force']:.0e}N_PF{params['point_force_mag']:.0e}N{params['TYPE_PF']}_k{params['k']:.0e}_kt{params['kt']:.0e}_fps{params['rendering_fps']}_stepskip{step_skip:.0f}"
    name = f"{params['scaling_type']}_FiberSim_{params['num_horizontal_threads']+params['num_vertical_threads']}rods_{suffix}"

    file_path = f"{params['loc']}{name}"
    n_files = 30+1

    start = 0
    step = np.rint(params['rendering_fps']/params['sample_freq_pf']).astype(int)
    input_data, output_data, time_data = load_simulation_data(file_path, n_files, start, params['num_horizontal_threads'], params['num_vertical_threads'], step)

    input_data = np.hstack([input_data[i] for i in range(len(input_data))])
    output_data = np.vstack([output_data[i] for i in range(len(output_data))])
    time_data = np.hstack([time_data[i] for i in range(len(time_data))])

    # Nonlinearity testing
    leg_max_order = 10
    regressor = "Rid"
    test_size = 0.75
    alpha = 1
    leg_capacity_train_list, leg_capacity_test_list = nonlinearity_testing(input_data, output_data, leg_max_order, regressor, test_size, alpha)
    
    # Memory testing
    max_time_back_seconds = 10
    max_time_back = params['sample_freq_pf']*max_time_back_seconds
    regressor = "Rid"
    test_size = 0.75
    alpha = 1
    mem_capacity_train_list, mem_capacity_test_list = memory_testing(input_data, output_data, max_time_back, regressor, test_size, alpha)

    # Plotting
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(np.linspace(0, leg_max_order, leg_max_order+1), leg_capacity_train_list, '-o', label='Training Capacity')
    plt.plot(np.linspace(0, leg_max_order, leg_max_order+1), leg_capacity_test_list, '-o', label='Testing Capacity')
    plt.xlabel('Legendre Polynomial Order')
    plt.ylabel('Capacity')
    plt.title('Nonlinearity')
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(np.linspace(0, max_time_back_seconds, max_time_back+1), mem_capacity_train_list, '-o', label='Training Capacity')
    plt.plot(np.linspace(0, max_time_back_seconds, max_time_back+1), mem_capacity_test_list, '-o', label='Testing Capacity')
    plt.xlabel('Seconds in the Past')
    plt.ylabel('Capacity')
    plt.title('Memory')
    plt.legend()
    plt.grid()

    plt.show()  # Show the plots





    