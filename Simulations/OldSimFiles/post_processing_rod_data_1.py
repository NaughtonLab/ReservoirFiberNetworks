import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.special import legendre


def post_processing_rod_data(name, sample_freq):

    file = f"processing_data/{name}"
    try:
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
    except:
        print('--file not found--')
    
    test_size = 0.75

    rods_history, force_profile = data

    force_profile = np.array(force_profile)
    input = force_profile # this is u(t) from the paper
    time = np.array(rods_history[0]["time"])

    total_num_rods = len(rods_history)
    num_horizontal_threads = int(total_num_rods/2)
    num_vertical_threads = total_num_rods - num_horizontal_threads

    rod_pos = []
    rod_dir = []

    for i in range(total_num_rods):
        rod_pos.append(np.array(rods_history[i]["position"]))
        rod_dir.append(np.array(rods_history[i]["directors"]))

    n_elem = rod_pos[0].shape[2]-1

    vert_connect_idx = np.linspace(0, n_elem+1, num_horizontal_threads+2)
    vert_connect_idx = vert_connect_idx[1:-1]
    hor_connect_idx = np.linspace(0, n_elem+1, num_vertical_threads+2)
    hor_connect_idx = hor_connect_idx[1:-1]

    num_connection_nodes = num_horizontal_threads * num_vertical_threads
    connection_nodes = np.hstack(rod_pos[i][..., int(hor_connect_idx[j])][..., 0:2] for i in range(num_horizontal_threads) for j in range(num_horizontal_threads))

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

    start = 250 # to get rid of transient behavior at the beginning
    stop = len(force_profile)

    scaler = preprocessing.StandardScaler().fit(output)
    output = scaler.transform(output)

    output = output[start:stop] # displacements - x(t)
    input = input[start:stop] # force - u(t)
    time = time[start:stop]

    input_base = input[:,None]
    # print(input_base.shape)

    capacity_list = []
    capacity_list_previous_inp = []

    for i in range(num_outputs):
        x = output
        polynomial = legendre(i)
        z = polynomial(input) # legendre polynomial of force
        # print(i)

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # ax1.plot(time, z)
        # ax1.set_title("legendre")
        # ax1.grid(True)

        # ax2.plot(time, output[..., 2*i:2*i+2])
        # ax2.set_title("displacement")
        # ax2.grid(True)
        # plt.show()

        # The profile of legendre polynomial of order 1 of force exactly matches the profile of Y displacement of h1a, h1b and h1c points
        # Although the magnitude of these profiles is different, why does it match?

        clf = LinearRegression()

        x_train, x_test, z_train, z_test = train_test_split(x, z, 
                                                            test_size=test_size,
                                                            random_state=42,
                                                            shuffle=False)
        
        # Here we are fitting the output of the reservoir (displacements) to the input (legendre polynomial of force)
        clf.fit(x_train, z_train)
        # z_hat = sum(W_ix_i) where x_i is the output of the reservoir, i.e., displacements
        z_hat = clf.predict(x_test)

        MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
        z2 = (1/len(z_test)) * np.sum((z_test)**2)
        C = 1 - MSE/z2

        capacity_list.append(C)

        z_previous = input_base
        clf.fit(z_previous, z)
        z_hat = clf.predict(z_previous)
        MSE = (1/len(z)) * np.sum((z_hat - z)**2)
        z2 = (1/len(z)) * np.sum((z)**2)
        C_prev = 1 - MSE/z2
        capacity_list_previous_inp.append(C_prev)

    plt.subplot(2,1,1)
    plt.plot(capacity_list_previous_inp[:10], 'ko-', label=None)
    plt.plot(capacity_list[:10], 'o-')#, label=f"Damping: {damping:.0f}")
    plt.ylim([0,1.05])
    # plt.legend(fontsize = 8)
    # plt.show()

    capacity_list = []
    capacity_list_previous_inp = []

    max_time = int(250 / sample_freq * 5)
    skip = 2

    for n in range(1, max_time, skip):
        x = output[n:] * 1.0
        z = input[:-n] * 1.0
        time_ = time[n:]

        clf = LinearRegression()

        x_train, x_test, z_train, z_test = train_test_split(x, z, 
                                                            test_size=test_size,
                                                            random_state=42,
                                                            shuffle=False)
        
        
        clf.fit(x_train, z_train)
        
        z_hat = clf.predict(x_test)

        MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
        z2 = (1/len(z_test)) * np.sum((z_test)**2)
        C = 1 - MSE/z2

        capacity_list.append(C)

        z_previous = input[n:] * 1.0
        z_previous = z_previous[:, None]
        clf.fit(z_previous, z)
        z_hat = clf.predict(z_previous)
        MSE = (1/len(z)) * np.sum((z_hat - z)**2)
        z2 = (1/len(z)) * np.sum((z)**2)
        C_prev = 1 - MSE/z2
        capacity_list_previous_inp.append(C_prev)

    plt.subplot(2,1,2)
    plt.plot(time[:max_time:skip]-1, capacity_list_previous_inp, '-k')
    plt.plot(time[:max_time:skip]-1, capacity_list)
    plt.ylim([0,1.05])
    # plt.legend(fontsize = 8)
    plt.show()

    
if __name__ == '__main__':
    max_episode_final_time = 10
    sample_freq = 5
    damping_constant = 10
    tf = 1e4
    point_force_mag = -2e5
    k = 1e9
    kt = 1e9
    suffix = f'sweep_{max_episode_final_time:.0f}sec_{sample_freq:.0f}hz_d{damping_constant:.0f}_tension{tf:.0e}_pointforce{point_force_mag:.0e}_k{k:.0e}_kt{kt:.0e}'
    
    num_horizontal_threads = 2
    num_vertical_threads = num_horizontal_threads

    name = f"{num_horizontal_threads+num_vertical_threads}rods_{suffix}.pickle"
    print(name)

    post_processing_rod_data(name=name, sample_freq=sample_freq)

# 8 rods -2e6 not quite good
# 8 rods -2e5 could be considered
# 8 rods -5e6 k 2e9 moderate
# 8 rods -5e6 k 5e9 not good -- k higher than 1e9 is not giving good results


