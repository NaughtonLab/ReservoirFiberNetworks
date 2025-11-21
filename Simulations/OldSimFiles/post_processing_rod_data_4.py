import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.special import legendre

class post_processing():

    def __init__(self, **kwargs):

        self.n_files = kwargs.get("n_files", 1)
        self.start = kwargs.get("start", 250)
        self.test_size = kwargs. get("test_size", 0.75)
        self.regressor = kwargs.get("regressor", "Lin")
        self.plot = kwargs.get("plot", False)
        self.objective = kwargs.get("objective", "leg")

    def get_data_from_file(self, filename):

        # file = f"processing_data/{filename}.pickle"
        file = f"./{filename}.pickle"

        print(f"{file}")

        try:
            with open(file, 'rb') as handle:
                data = pickle.load(handle)
        except:
            print('--file not found--')

        rods_history, force_profile = data

        force_profile = np.array(force_profile)
        input = force_profile # this is u(t) from the paper
        time = np.array(rods_history[0]["time"])
        print("time", time.shape)

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
                print("conn node shape", rod_pos[i][..., int(hor_connect_idx[j])][..., 0:2].shape)
                connection_nodes.append(rod_pos[i][..., int(hor_connect_idx[j])][..., 0:2])

        connection_nodes = np.hstack([connection_nodes[i] for i in range(len(connection_nodes))])
        print("conn node shape 2", connection_nodes.shape)

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

        print("output shape", output.shape)
        print(output[0:1])

        # WHY DO THIS IF WE ARE GOING TO USE STANDARDSCALER LATER??
        output /= np.mean(output, axis=0) # mean is calculated along the rows i.e., the number of columns stay the same
        output /= np.std(output, axis=0)

        start = self.start # to get rid of transient behavior at the beginning
        stop = len(force_profile)
        
        scaler = preprocessing.StandardScaler().fit(output)
        output = scaler.transform(output)

        output = output[start:stop] # displacements - x(t)
        input = input[start:stop] # force - u(t)
        time = time[start:stop]

        return input, output, time
    
    def post_processing_rod_data(self, name):
        
        if self.n_files == 1:
            filename = name
            input, output, time = self.get_data_from_file(filename)
        else:
            input = []
            output = []
            time = []

            for k in range(1, self.n_files+1):
                filename = f"{name}_{k}"
                ip, op, t = self.get_data_from_file(filename)

                input.append(ip)
                output.append(op)
                time.append(t)

            input = np.hstack([input[i] for i in range(len(input))])
            output = np.vstack([output[i] for i in range(len(output))])
            time = np.hstack([time[i] for i in range(len(time))])

        if self.regressor == "Lin":
            ### Linear Regression
            clf = LinearRegression()
        elif self.regressor == "Rid":
            ### Ridge Regression
            clf = Ridge()
        else:
            print("Please specify the regressor")

        if self.objective == "memory":
            '''Case 1: Reproduce n-steps old input'''
            # z(t) = u(t) --> x = output; z = input
            # z(t) = u(t-1*dt) --> x = output[1:]; z = input[:-1]
            # z(t) = u(t-n*dt) --> x = output[n:]; z = input[:-n]
            capacity_list_1_train = []
            capacity_list_1_test = []
            n_max = 100
            for n in range(n_max+1):
                x = output[n:]
                if n == 0:
                    z = input
                else:
                    z = input[:-n]

                x_train, x_test, z_train, z_test = train_test_split(x, z, 
                                                                    test_size=self.test_size,
                                                                    random_state=42,
                                                                    shuffle=False)
                    
                
                clf.fit(x_train, z_train)

                # Training capacity
                z_hat_train = clf.predict(x_train)

                MSE = (1/len(z_train)) * np.sum((z_hat_train - z_train)**2)
                z2 = (1/len(z_train)) * np.sum((z_train)**2)
                C = 1 - MSE/z2

                capacity_list_1_train.append(C)

                # Testing capacity
                z_hat = clf.predict(x_test)

                MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
                z2 = (1/len(z_test)) * np.sum((z_test)**2)
                C = 1 - MSE/z2

                capacity_list_1_test.append(C)

            '''Case 2: Reproduce product of current and n-steps old input'''
            # # z(t) = u(t)*u(t-1) --> x = output[1:]; z = input[1:]*input[:-1]
            # capacity_list_2_train = []
            # capacity_list_2_test = []
            # n_max = 100
            # for n in range(1, n_max+1):
                # x = output[n:]
                # z = input[n:]*input[:-n]

                # x_train, x_test, z_train, z_test = train_test_split(x, z, 
                #                                                     test_size=self.test_size,
                #                                                     random_state=42,
                #                                                     shuffle=False)
                
                # clf.fit(x_train, z_train)

                # # Training capacity
                # z_hat_train = clf.predict(x_train)

                # MSE = (1/len(z_train)) * np.sum((z_hat_train - z_train)**2)
                # z2 = (1/len(z_train)) * np.sum((z_train)**2)
                # C = 1 - MSE/z2

                # capacity_list_2_train.append(C)

                # # Testing capacity
                # z_hat = clf.predict(x_test)

                # MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
                # z2 = (1/len(z_test)) * np.sum((z_test)**2)
                # C = 1 - MSE/z2

                # capacity_list_2_test.append(C)

            '''Case 3: Reproduce product of current, n-steps and n-1-steps old input'''
            # z(t) = u(t)*u(t-1)*u(t-2) --> x = output[2:]; z = input[2:]*input[1:-1]*input[:-2]
            # capacity_list_3_train = []
            # capacity_list_3_test = []
            # n_max = 100
            # for n in range(1, n_max+1):
                # x = output[n:]
                # z = input[n:]*input[n-1:-1]*input[:-n]

                # x_train, x_test, z_train, z_test = train_test_split(x, z, 
                #                                                     test_size=self.test_size,
                #                                                     random_state=42,
                #                                                     shuffle=False)
                
                # clf.fit(x_train, z_train)

                # # Training capacity
                # z_hat_train = clf.predict(x_train)

                # MSE = (1/len(z_train)) * np.sum((z_hat_train - z_train)**2)
                # z2 = (1/len(z_train)) * np.sum((z_train)**2)
                # C = 1 - MSE/z2

                # capacity_list_3_train.append(C)

                # # Testing capacity
                # z_hat = clf.predict(x_test)

                # MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
                # z2 = (1/len(z_test)) * np.sum((z_test)**2)
                # C = 1 - MSE/z2

                # capacity_list_3_test.append(C)

            capacity_train = sum(capacity_list_1_train) #[capacity_list_1_train, capacity_list_2_train, capacity_list_3_train]
            capacity_test = sum(capacity_list_1_test) #[capacity_list_1_test, capacity_list_2_test, capacity_list_3_test]

            return capacity_train, capacity_test

        elif self.objective == "leg":
            '''Case 1: Reproduce legendre polynomial of input'''
            # z(t) = n-legendre(u(t)) --> x = output; polynomial = legendre(n); z = polynomial(input)
            # z(t) = n-legendre(u(t-h)) --> x = output[h:]; polynomial = legendre(n); z = polynomial(input[:-h])
            capacity_list_1_train = []
            capacity_list_1_test = []
            n_max = 15
            for n in range(n_max+1):
                x = output
                polynomial = legendre(n)
                z = polynomial(input)

                x_train, x_test, z_train, z_test = train_test_split(x, z, 
                                                                    test_size=self.test_size,
                                                                    random_state=42,
                                                                    shuffle=False)

                clf.fit(x_train, z_train)

                # Training capacity
                z_hat_train = clf.predict(x_train)

                MSE = (1/len(z_train)) * np.sum((z_hat_train - z_train)**2)
                z2 = (1/len(z_train)) * np.sum((z_train)**2)
                C = 1 - MSE/z2

                capacity_list_1_train.append(C)

                # Testing capacity
                z_hat = clf.predict(x_test)

                MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
                z2 = (1/len(z_test)) * np.sum((z_test)**2)
                C = 1 - MSE/z2

                capacity_list_1_test.append(C)

            capacity_train = sum(capacity_list_1_train)
            capacity_test = sum(capacity_list_1_test)

            return capacity_train, capacity_test
        
        elif self.objective == "nonlin":
            '''Case 1: Reproduce legendre polynomial of input'''
            # z(t) = n-legendre(u(t)) --> x = output; polynomial = legendre(n); z = polynomial(input)
            # z(t) = n-legendre(u(t-h)) --> x = output[h:]; polynomial = legendre(n); z = polynomial(input[:-h])
            capacity_list_1_train = []
            capacity_list_1_test = []
            n_max = 15
            for n in range(n_max+1):
                x = output
                polynomial = legendre(n)
                z = polynomial(input)

                x_train, x_test, z_train, z_test = train_test_split(x, z, 
                                                                    test_size=self.test_size,
                                                                    random_state=42,
                                                                    shuffle=False)

                clf.fit(x_train, z_train)

                # Training capacity
                z_hat_train = clf.predict(x_train)

                MSE = (1/len(z_train)) * np.sum((z_hat_train - z_train)**2)
                z2 = (1/len(z_train)) * np.sum((z_train)**2)
                C = 1 - MSE/z2

                capacity_list_1_train.append(C)

                # Testing capacity
                z_hat = clf.predict(x_test)

                MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
                z2 = (1/len(z_test)) * np.sum((z_test)**2)
                C = 1 - MSE/z2

                capacity_list_1_test.append(C)

            '''Case 2: Reproduce powers of the input'''
            # z(t) = u(t)^2 --> x = output; z = input**2
            # z(t) = u(t)^3 --> x = output; z = input**3
            capacity_list_2_train = []
            capacity_list_2_test = []
            n_max = 10
            for n in range(1, n_max+1):
                x = output
                z = input**n

                x_train, x_test, z_train, z_test = train_test_split(x, z, 
                                                                    test_size=self.test_size,
                                                                    random_state=42,
                                                                    shuffle=False)
                
                clf.fit(x_train, z_train)

                # Training capacity
                z_hat_train = clf.predict(x_train)

                MSE = (1/len(z_train)) * np.sum((z_hat_train - z_train)**2)
                z2 = (1/len(z_train)) * np.sum((z_train)**2)
                C = 1 - MSE/z2

                capacity_list_2_train.append(C)

                # Testing capacity
                z_hat = clf.predict(x_test)

                MSE = (1/len(z_test)) * np.sum((z_hat - z_test)**2)
                z2 = (1/len(z_test)) * np.sum((z_test)**2)
                C = 1 - MSE/z2

                capacity_list_2_test.append(C)

            capacity_train = sum(capacity_list_1_train) + sum(capacity_list_2_train)
            capacity_test = sum(capacity_list_1_test) + sum(capacity_list_2_test)

            return capacity_train, capacity_test
        
        else:
            print("Please specify the objective")

        if self.plot:
            plt.figure(figsize=(25, 20)) # width by height
            # plt.tight_layout(h_pad=10, w_pad=10)
            plt.rcParams.update({"font.size": 17})

            plt.subplot(2, 2, 1)
            plt.plot(capacity_list_1_train, '-o', label = 'Linear')
            # plt.plot(capacity_list_1_test_RidReg, '-o', label = 'Ridge')
            plt.xlabel("Number of Timesteps in the past")
            plt.ylabel("Capacity")
            plt.grid(True)
            plt.legend()
            plt.xlim([0, 60])
            plt.ylim([0,1.05])
            plt.title(f"Memory capacity on training set")

            plt.subplot(2, 2, 3)
            plt.plot(capacity_list_1_test, '-o', label = 'Linear')
            # plt.plot(capacity_list_1_test_RidReg, '-o', label = 'Ridge')
            plt.xlabel("Number of Timesteps in the past")
            plt.ylabel("Capacity")
            plt.grid(True)
            plt.legend()
            plt.xlim([0, 60])
            plt.ylim([0,1.05])
            plt.title(f"Memory capacity on testing set")

            plt.subplot(2, 2, 2)
            plt.plot(capacity_list_2_train, '-o', label = 'Linear')
            # plt.plot(capacity_list_2_train_RidReg, '-o', label = 'Ridge')
            plt.xlabel("Order of Legendre Polynomial")
            plt.ylabel("Capacity")
            plt.grid(True)
            plt.legend()
            plt.ylim([0,1.05])
            plt.title(f"Capacity to reproduce Legendre Polynomial on training set")

            plt.subplot(2, 2, 4)
            plt.plot(capacity_list_2_test, '-o', label = 'Linear')
            # plt.plot(capacity_list_6_test_RidReg, '-o', label = 'Ridge')
            plt.xlabel("Order of Legendre Polynomial")
            plt.ylabel("Capacity")
            plt.grid(True)
            plt.legend()
            plt.ylim([0,1.05])
            plt.title(f"Capacity to reproduce Legendre Polynomial on training set")

            # plt.savefig(f'processing_data/CAPACITY_{name}.png')


if __name__=="__main__":
    p = post_processing()
    p.post_processing_rod_data(name="4rods_sweep_100sec_5hz_d10_tension1e+04_pointforce-2e+05_k1e+09_kt1e+09_6")
