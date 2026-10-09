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

        # Testing
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 1)
        y_test_pred[idx] = np.mean(y)
        y_test[idx, 0] = np.mean(y)

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

def memory_testing(input, output, max_timesteps_back, regressor, test_size, alpha, sample_freq):
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

        # Testing
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 1)
        y_test_pred[idx] = np.mean(y)
        y_test[idx, 0] = np.mean(y)

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

            # Testing
            y_test_pred = clf.predict(x_test)

            idx = np.where(abs(y_test_pred) > 1)
            y_test_pred[idx] = np.mean(y)
            y_test[idx, 0] = np.mean(y)

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
    num_sides_polygon = 6
    polygon_diameter = 600
    dx = 10

    every_x_frame = 1
    sample_freq = np.rint(250/every_x_frame).astype(int)

    data = np.load(f"./polygon_data_4.npz", allow_pickle=True)

    input_data = data['input_data'][:, np.newaxis]
    
    input_data = -1 + (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data)) * (1 - (-1))
    # plt.figure(figsize=(10, 5))
    # plt.plot(input_data, label= 'input')
    # # leg = legendre(5)
    # # # plt.scatter(input_data, leg(input_data), label='Legendre 5')
    # plt.legend()
    # plt.show()

    output_data = data['output_data']
    # plt.figure(figsize=(10, 5))
    # plt.plot(output_data, label= 'output')
    # plt.show()
    time_data = data['time_data']

    # Nonlinearity testing
    leg_max_order = 10
    regressor = "Rid"
    test_size = 0.5
    alpha = 20
    leg_capacity_train_list, leg_capacity_test_list, leg_R2_train_list, leg_R2_test_list = nonlinearity_testing(input_data, output_data, leg_max_order, regressor, test_size, alpha)
    
    # Memory testing
    max_time_back_seconds = 1
    max_timesteps_back = np.rint(sample_freq*max_time_back_seconds).astype(int)
    # max_timesteps_back = 10
    # max_time_back_seconds = np.rint(max_timesteps_back/sample_freq).astype(int)
    regressor = "Rid"
    test_size = 0.5
    alpha = 20
    mem_capacity_train_list, mem_capacity_test_list, mem_R2_train_list, mem_R2_test_list = memory_testing(input_data, output_data, max_timesteps_back, regressor, test_size, alpha, sample_freq)

    np.savez("./polygon_eval_4.npz", legendre_R2=(leg_R2_train_list, leg_R2_test_list), legendre_cap=(leg_capacity_train_list, leg_capacity_test_list),
                memory_R2=(mem_R2_train_list, mem_R2_test_list), memory_cap=(mem_capacity_train_list, mem_capacity_test_list))

    # Plotting Legendre and Memory
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(np.linspace(1, leg_max_order, leg_max_order), leg_capacity_train_list, '-o', label='Training')
    plt.plot(np.linspace(1, leg_max_order, leg_max_order), leg_capacity_test_list, '-o', label='Testing')
    plt.xlabel('Legendre Polynomial Order')
    plt.ylabel('Capacity')
    plt.title('Nonlinearity Capacity')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid()
    
    plt.subplot(122)
    plt.plot(np.linspace(1, leg_max_order, leg_max_order), leg_R2_train_list, '-o', label='Training')
    plt.plot(np.linspace(1, leg_max_order, leg_max_order), leg_R2_test_list, '-o', label='Testing')
    plt.xlabel('Legendre Polynomial Order')
    plt.ylabel('R2 score')
    plt.title('Nonlinearity R2 score')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid()

    plt.savefig(f"./polygon_legendre_eval_4.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_capacity_train_list, '-o', label='Training')
    plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_capacity_test_list, '-o', label='Testing')
    plt.xlabel('Seconds in the Past')
    plt.ylabel('Capacity')
    plt.title('Memory Capacity')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_R2_train_list, '-o', label='Training')
    plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_R2_test_list, '-o', label='Testing')
    plt.xlabel('Seconds in the Past')
    plt.ylabel('R2 score')
    plt.title('Memory R2 score')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid()
    
    # plt.show()  # Show the plots
    plt.savefig(f"./polygon_memory_eval_4.png", dpi=300, bbox_inches='tight')
    plt.close()