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

matplotlib.rc('pdf', fonttype=42)

def nonlinearity_testing(input_sig, output, leg_max_order, regressor, test_size, alpha, net_size):
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
        y = leg(input_sig)
        shape_input = input_sig.shape
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

        pred = np.concatenate((y_train_pred, y_test_pred), axis=0)

        if n < 5:
            plt.figure(figsize=(6, 6))
            x_axis = np.linspace(-1, 1, y.shape[0])
            plt.plot(x_axis, leg(x_axis), color='k', label='True')
            plt.scatter(input_sig, pred, s=10, color='tab:orange', label='Predicted', zorder=2)
            plt.title(f'Legendre Polynomial Order: {n}')
            plt.legend()
            plt.ylim(-1.25, 1.25)
            plt.savefig(f'Fig1_leg_{n}_{net_size}.pdf', dpi=300) 
            plt.close()   

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def memory_testing(input_sig, output, max_timesteps_back, regressor, test_size, alpha, net_size):
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
            y = input_sig
        else:
            y = input_sig[:-n]

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

        n_sec = n/250
        if n%25 == 0 and n_sec < 0.5:
            num_steps = int(250*20)
            time_axis = np.linspace(0, len(y_test)/250, len(y_test))
            plt.figure(figsize=(6, 6))
            plt.plot(time_axis, input_sig[-1-len(y_test):-1], '-', lw=4, label=f'Actual input at {n_sec}')
            plt.plot(time_axis, y_test, '-', lw=2.5, label='Target')
            plt.plot(time_axis, y_test_pred, '-', lw=1, label='Predicted')
            plt.title(f'Recall Time {n_sec} s')
            plt.ylim(-1.25, 1.25)
            plt.legend()
            plt.savefig(f'Fig1_mem_{n_sec}_{net_size}.pdf', dpi=300)    
            plt.close()

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

        # print("memory", n, R2_train, R2_test, capacity_train, capacity_test)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

if __name__ == "__main__":

    fps = 250
    folder = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(folder, 'Data', '')

    csv_name = os.path.join(folder, 'IncDensity')

    grid = np.load(os.path.join(folder, 'density_list.npz'), allow_pickle=True)
    grid = grid['grid']
    
    regressor = "Rid"
    test_size = 0.25
    alpha = 1e-2    
    
    leg_max_order = 10
    max_time_back_seconds = 1
    max_timesteps_back = np.rint(fps*max_time_back_seconds).astype(int)

    idx_list = [i for i in range(6)]

    for idx in idx_list:

        grid_data = grid[idx]
        num_horizontal_threads = int(grid_data[0])
        num_vertical_threads = num_horizontal_threads
        point_force_mag = grid_data[2]
        spacing = grid_data[1]
        thread_length = 500e-3

        data = np.load(f"{path}{idx}_eval.npz", allow_pickle=True)
        input_data = data['input_data']
        output_data = data['output_data']
        time_data = data['time_data']

        if not np.isnan(input_data).any():
            print(idx)

            input_data = -1 + (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data)) * (1 - (-1))

            print(input_data.shape, output_data.shape)

            # Nonlinearity testing
            leg_capacity_train_list, leg_capacity_test_list, leg_R2_train_list, leg_R2_test_list = nonlinearity_testing(input_data, output_data, leg_max_order, regressor, test_size, alpha, num_horizontal_threads)
            
            # Memory testing
            mem_capacity_train_list, mem_capacity_test_list, mem_R2_train_list, mem_R2_test_list = memory_testing(input_data, output_data, max_timesteps_back, regressor, test_size, alpha, num_horizontal_threads)
