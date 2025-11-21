import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.special import legendre

def get_data_from_file(name, k):
    try:
        file = f"processing_data/{name}_{k}.pickle"
        print(f"{name}_{k}")

        with open(file, 'rb') as handle:
            data = pickle.load(handle)
    except:
        print('--file not found--')

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

    start = 250 # to get rid of transient behavior at the beginning
    stop = len(force_profile)

    scaler = preprocessing.StandardScaler().fit(output)
    output = scaler.transform(output)

    output = output[start:stop] # displacements - x(t)
    input = input[start:stop] # force - u(t)
    time = time[start:stop]

    return input, output, time


def post_processing_rod_data(name, sample_freq):

    n_files = 5
    capacity = np.empty((6, n_files), dtype=list)
    test_size = 0.75

    plt.figure(figsize=(25, 20)) # width by height
    # plt.tight_layout(h_pad=10, w_pad=10)
    plt.rcParams.update({"font.size": 17})

    input_train = []
    output_train = []
    time_train = []

    input_test = []
    output_test = []
    time_test = []

    for k in range(1, n_files):

        if k == 4:
            l = 5
        else:
            l = k
        
        input, output, time = get_data_from_file(name, l)

        if input.shape[0] > 2250:
            input = input[:-1]
            output = output[:-1, :]
            time = time[:-1]

        input_train.append(input)
        output_train.append(output)
        time_train.append(time + (k-1)*10)

        # print(f"Done file {k}")

    input_train = np.hstack([input_train[i] for i in range(len(input_train))])
    output_train = np.vstack([output_train[i] for i in range(len(output_train))])
    time_train = np.hstack([time_train[i] for i in range(len(time_train))])

    input_test, output_test, time_test = get_data_from_file(name, 4)

    '''Case 1: Reproduce n-steps old input'''
    # z(t) = u(t) --> x = output; z = input
    # z(t) = u(t-1*dt) --> x = output[1:]; z = input[:-1]
    # z(t) = u(t-n*dt) --> x = output[n:]; z = input[:-n]
    capacity_list_1 = []
    capacity_list_1_train = []
    n_max = 100
    for n in range(n_max+1):
        x_train = output_train[n:]
        x_test = output_test[n:]
        if n == 0:
            z_train = input_train
            z_test = input_test
        else:
            z_train = input_train[:-n]
            z_test = input_test[:-n]

        # x_train, x_test, z_train, z_test = train_test_split(x, z, 
        #                                                     test_size=test_size,
        #                                                     random_state=42,
        #                                                     shuffle=False)
        
        clf = LinearRegression()
        clf.fit(x_train, z_train)
        z_hat_train = clf.predict(x_train)
        MSE = (1/len(z_train)) * np.sum((z_hat_train - z_train)**2)
        z2 = (1/len(z_train)) * np.sum((z_train)**2)
        C = 1 - MSE/z2

        capacity_list_1_train.append(C)

        z_hat = clf.predict(x_test)

        MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
        z2 = (1/len(z_test)) * np.sum((z_test)**2)
        C = 1 - MSE/z2

        capacity_list_1.append(C)

    # capacity[0, k-1] = capacity_list_1

    '''Case 2: Reproduce product of current and n-steps old input'''
    # z(t) = u(t)*u(t-1) --> x = output[1:]; z = input[1:]*input[:-1]
    capacity_list_2 = []
    n_max = 100
    for n in range(1, n_max+1):
        x_train = output_train[n:]
        z_train = input_train[n:]*input_train[:-n]

        x_test = output_test[n:]
        z_test = input_test[n:]*input_test[:-n]

        # x_train, x_test, z_train, z_test = train_test_split(x, z, 
        #                                                     test_size=test_size,
        #                                                     random_state=42,
        #                                                     shuffle=False)
        
        clf = LinearRegression()
        clf.fit(x_train, z_train)
        z_hat = clf.predict(x_test)

        MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
        z2 = (1/len(z_test)) * np.sum((z_test)**2)
        C = 1 - MSE/z2

        capacity_list_2.append(C)

    # capacity[1, k-1] = capacity_list_2

    '''Case 3: Reproduce product of current, n-steps and n-1-steps old input'''
    # z(t) = u(t)*u(t-1)*u(t-2) --> x = output[2:]; z = input[2:]*input[1:-1]*input[:-2]
    capacity_list_3 = []
    n_max = 100
    for n in range(2, n_max+1):
        x_train = output_train[n:]
        z_train = input_train[n:]*input_train[n-1:-1]*input_train[:-n]

        x_test = output_test[n:]
        z_test = input_test[n:]*input_test[n-1:-1]*input_test[:-n]

        # x_train, x_test, z_train, z_test = train_test_split(x, z, 
        #                                                     test_size=test_size,
        #                                                     random_state=42,
        #                                                     shuffle=False)
        
        clf = LinearRegression()
        clf.fit(x_train, z_train)
        z_hat = clf.predict(x_test)

        MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
        z2 = (1/len(z_test)) * np.sum((z_test)**2)
        C = 1 - MSE/z2

        capacity_list_3.append(C)

    # capacity[2, k-1] = capacity_list_3

    '''Case 4: Reproduce powers of the input'''
    # z(t) = u(t)^2 --> x = output; z = input**2
    # z(t) = u(t)^3 --> x = output; z = input**3
    capacity_list_4 = []
    n_max = 10
    for n in range(1, n_max+1):
        x_train = output_train
        z_train = input_train**n

        x_test = output_test
        z_test = input_test**n

        # x_train, x_test, z_train, z_test = train_test_split(x, z, 
        #                                                     test_size=test_size,
        #                                                     random_state=42,
        #                                                     shuffle=False)
        
        clf = LinearRegression()
        clf.fit(x_train, z_train)
        z_hat = clf.predict(x_test)

        MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
        z2 = (1/len(z_test)) * np.sum((z_test)**2)
        C = 1 - MSE/z2

        capacity_list_4.append(C)

    # capacity[3, k-1] = capacity_list_4

    '''Case 5: Reproduce exponential function of the input'''
    # z(t) = sin(u(t)) --> x = output; z = np.sin(input)
    # z(t) = sin(u(t))*cos(u(t)) --> x = output; z = np.sin(input)*np.cos(input)
    capacity_list_5 = []
    n_max = 50
    for n in range(n_max+1):
        x_train = output_train
        z_train = np.exp(n*input_train)

        x_test = output_test
        z_test = np.exp(n*input_test)

        # x_train, x_test, z_train, z_test = train_test_split(x, z, 
        #                                                     test_size=test_size,
        #                                                     random_state=42,
        #                                                     shuffle=False)
        
        clf = LinearRegression()
        clf.fit(x_train, z_train)
        z_hat = clf.predict(x_test)

        MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
        z2 = (1/len(z_test)) * np.sum((z_test)**2)
        C = 1 - MSE/z2

        capacity_list_5.append(C)

    # capacity[4, k-1] = capacity_list_5
    
    '''Case 6: Reproduce legendre polynomial of input'''
    # z(t) = n-legendre(u(t)) --> x = output; polynomial = legendre(n); z = polynomial(input)
    # z(t) = n-legendre(u(t-h)) --> x = output[h:]; polynomial = legendre(n); z = polynomial(input[:-h])
    capacity_list_6 = []
    capacity_list_6_train = []
    n_max = 15
    for n in range(n_max+1):
        x_train = output_train
        polynomial = legendre(n)
        z_train = polynomial(input_train)

        x_test = output_test
        polynomial = legendre(n)
        z_test = polynomial(input_test)

        # x_train, x_test, z_train, z_test = train_test_split(x, z, 
        #                                                     test_size=test_size,
        #                                                     random_state=42,
        #                                                     shuffle=False)
        
        clf = LinearRegression()
        clf.fit(x_train, z_train)

        z_hat_train = clf.predict(x_train)
        MSE = (1/len(z_train)) * np.sum((z_hat_train - z_train)**2)
        z2 = (1/len(z_train)) * np.sum((z_train)**2)
        C = 1 - MSE/z2

        capacity_list_6_train.append(C)

        z_hat = clf.predict(x_test)

        MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
        z2 = (1/len(z_test)) * np.sum((z_test)**2)
        C = 1 - MSE/z2

        capacity_list_6.append(C)

    # capacity[5, k-1] = capacity_list_6
        
    plt.subplot(2, 2, 1)
    plt.plot(capacity_list_1_train, '-o')#, label = f"{k}")
    # plt.plot(capacity[0, 1], '-o')#, label = 2)
    # plt.plot(capacity[0, 2], '-o')#, label = 3)
    plt.xlabel("n-timesteps in the past")
    plt.ylabel("Capacity")
    plt.grid(True)
    # plt.legend()
    plt.xlim([0, 60])
    plt.ylim([0,1.05])
    plt.title(f"Memory capacity on training set")

    plt.subplot(2, 2, 3)
    plt.plot(capacity_list_1, '-o')#, label = f"{k}")
    # plt.plot(capacity[1, 1], '-o')#, label = 2)
    # plt.plot(capacity[1, 2], '-o')#, label = 3)
    plt.xlabel("n-timesteps in the past")
    plt.ylabel("Capacity")
    plt.grid(True)
    # plt.legend()
    plt.xlim([0, 60])
    plt.ylim([0,1.05])
    plt.title(f"Memory capacity on testing set")

    # plt.subplot(3, 2, 3)
    # plt.plot(capacity_list_3, '-o')#, label = f"{k}")
    # # plt.plot(capacity[2, 1], '-o')#, label = 2)
    # # plt.plot(capacity[2, 2], '-o')#, label = 3)
    # plt.xlabel("n")
    # plt.ylabel("Capacity")
    # plt.grid(True)
    # # plt.legend()
    # plt.xlim([0, 60])
    # plt.ylim([0,1.05])
    # plt.title(f"Capacity for $z(t) = u(t)u(t-(n-1)*dt)u(t-n*dt)$")

    # plt.subplot(3, 2, 4)
    # plt.plot(capacity_list_4, '-o')#, label = f"{k}")
    # # plt.plot(capacity[3, 1], '-o')#, label = 2)
    # # plt.plot(capacity[3, 2], '-o')#, label = 3)
    # plt.xlabel("n")
    # plt.ylabel("Capacity")
    # plt.grid(True)
    # # plt.legend()
    # plt.ylim([0,1.05])
    # plt.title(f"Capacity for $z(t) = u(t)^n$")

    plt.subplot(2, 2, 2)
    plt.plot(capacity_list_6_train, '-o')#, label = f"{k}")
    # plt.plot(capacity[4, 1], '-o')#, label = 2)
    # plt.plot(capacity[4, 2], '-o')#, label = 3)
    plt.xlabel("Order of Legendre Polynomial")
    plt.ylabel("Capacity")
    plt.grid(True)
    # plt.legend()
    plt.ylim([0,1.05])
    plt.title(f"Capacity to reproduce Legendre Polynomial on training set")

    plt.subplot(2, 2, 4)
    plt.plot(capacity_list_6, '-o')#, label = f"{k}")
    # plt.plot(capacity[5, 1], '-o')#, label = 2)
    # plt.plot(capacity[5, 2], '-o')#, label = 3)
    plt.xlabel("Order of Legendre Polynomial")
    plt.ylabel("Capacity")
    plt.grid(True)
    # plt.legend()
    plt.ylim([0,1.05])
    plt.title(f"Capacity to reproduce Legendre Polynomial on testing set")

    plt.savefig(f'processing_data/CAPACITY_{name}_1235train_4test_traintest.png')
    
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

    name = f"{num_horizontal_threads+num_vertical_threads}rods_{suffix}"
    # print(name)

    post_processing_rod_data(name=name, sample_freq=sample_freq)

# 8 rods -2e6 not quite good
# 8 rods -2e5 could be considered
# 8 rods -5e6 k 2e9 moderate
# 8 rods -5e6 k 5e9 not good -- k higher than 1e9 is not giving good results


