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
from sklearn.model_selection import KFold, TimeSeriesSplit
   
def nonlinearity_testing(input, output, leg_max_order, regressor, test_size, alpha, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=False)
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

        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
        kf_capacity_train_sum = 0
        kf_capacity_test_sum = 0
        kf_R2_train_sum = 0
        kf_R2_test_sum = 0
        for i, (train_idx, test_idx) in enumerate(kf.split(x)):
            # print(f"Fold {i}")
            x_train, x_test, y_train, y_test = x[train_idx, :], x[test_idx, :], y[train_idx, :], y[test_idx, :]

            # Training
            clf.fit(x_train, y_train)
            y_train_pred = clf.predict(x_train)
            
            idx = np.where(abs(y_train_pred) > 1)
            y_train_pred[idx] = np.mean(y)
            y_train[idx, 0] = np.mean(y)
            
            y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)

            MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
            kf_capacity_train = 1 - MSE/y2_train
            kf_R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

            # Testing
            y_test_pred = clf.predict(x_test)

            idx = np.where(abs(y_test_pred) > 1)
            y_test_pred[idx] = np.mean(y)
            y_test[idx, 0] = np.mean(y)

            y2_test = (1/len(y_test)) * np.sum((y_test-np.mean(y_test))**2)

            MSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
            kf_capacity_test = 1 - MSE/y2_test
            kf_R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

            if kf_R2_test < 0:
                kf_R2_test = 0
            if kf_R2_train < 0:
                kf_R2_train = 0
            if kf_capacity_test < 0:
                kf_capacity_test = 0
            if kf_capacity_train < 0:
                kf_capacity_train = 0

            kf_capacity_train_sum += kf_capacity_train
            kf_capacity_test_sum += kf_capacity_test
            kf_R2_train_sum += kf_R2_train
            kf_R2_test_sum += kf_R2_test

        capacity_train = kf_capacity_train_sum/n_splits
        capacity_test = kf_capacity_test_sum/n_splits
        R2_train = kf_R2_train_sum/n_splits
        R2_test = kf_R2_test_sum/n_splits

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

def memory_testing(input, output, max_timesteps_back, regressor, test_size, alpha, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
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
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
        tscv_capacity_train_sum = 0
        tscv_capacity_test_sum = 0
        tscv_R2_train_sum = 0
        tscv_R2_test_sum = 0
        for i, (train_idx, test_idx) in enumerate(tscv.split(x)):
            # print(f"Fold {i}")
            x_train, x_test, y_train, y_test = x[train_idx, :], x[test_idx, :], y[train_idx, :], y[test_idx, :]

            # Training
            clf.fit(x_train, y_train)
            y_train_pred = clf.predict(x_train)
            
            idx = np.where(abs(y_train_pred) > 1)
            y_train_pred[idx] = np.mean(y)
            y_train[idx, 0] = np.mean(y)
            
            y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)

            MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
            tscv_capacity_train = 1 - MSE/y2_train
            tscv_R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

            # Testing
            y_test_pred = clf.predict(x_test)

            idx = np.where(abs(y_test_pred) > 1)
            y_test_pred[idx] = np.mean(y)
            y_test[idx, 0] = np.mean(y)

            y2_test = (1/len(y_test)) * np.sum((y_test-np.mean(y_test))**2)

            MSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
            tscv_capacity_test = 1 - MSE/y2_test
            tscv_R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

            if tscv_R2_test < 0:
                tscv_R2_test = 0
            if tscv_R2_train < 0:
                tscv_R2_train = 0
            if tscv_capacity_test < 0:
                tscv_capacity_test = 0
            if tscv_capacity_train < 0:
                tscv_capacity_train = 0

            tscv_capacity_train_sum += tscv_capacity_train
            tscv_capacity_test_sum += tscv_capacity_test
            tscv_R2_train_sum += tscv_R2_train
            tscv_R2_test_sum += tscv_R2_test

        capacity_train = tscv_capacity_train_sum/n_splits
        capacity_test = tscv_capacity_test_sum/n_splits
        R2_train = tscv_R2_train_sum/n_splits
        R2_test = tscv_R2_test_sum/n_splits

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


if __name__ == "__main__":

    fps = 250
    n_splits = 10
    folder = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(folder, 'Data', '')

    csv_name = os.path.join(folder, f'IncDensity_{n_splits}_alphaCV')

    grid = np.load(os.path.join(folder, 'density_list.npz'), allow_pickle=True)
    grid = grid['grid']
    
    regressor = "Rid"
    test_size = 0.25
    alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]

    leg_max_order = 10
    max_time_back_seconds = 1
    max_timesteps_back = np.rint(fps*max_time_back_seconds).astype(int)

    idx_list = [i for i in range(6)]

    df = pd.DataFrame(columns = ['num_threads', 'alpha', 'nonlinearity train', 'nonlinearity test', 'memory train', 'memory test'])

    j = 0
    for idx in idx_list:
        # npsavez_dict = {}

        grid_data = grid[idx]
        num_horizontal_threads = int(grid_data[0])
        num_vertical_threads = num_horizontal_threads
        point_force_mag = grid_data[2]
        spacing = grid_data[1]
        thread_length = 500e-3
        
        data = np.load(f"{path}{idx}_eval_{n_splits}_alphaCV.npz", allow_pickle=True)
        data = data['npsavez_dict'].item()
        print(data.keys())
        # input_data = data['input_data']
        # output_data = data['output_data']
        # time_data = data['time_data']

        for alpha in alpha_list:

            metrics = data[f'{alpha}']

            leg_capacity_train_list = metrics[0]
            leg_capacity_test_list = metrics[1]
            mem_capacity_train_list = metrics[2]
            mem_capacity_test_list = metrics[3]

            onc_train = sum(leg_capacity_train_list)/len(leg_capacity_train_list)
            omc_train = sum(mem_capacity_train_list)/len(mem_capacity_train_list)

            onc_test = sum(leg_capacity_test_list)/len(leg_capacity_test_list)
            omc_test = sum(mem_capacity_test_list)/len(mem_capacity_test_list)

            # if not np.isnan(input_data).any():
            #     print(idx)

            #     input_data = -1 + (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data)) * (1 - (-1))

            #     print(input_data.shape, output_data.shape)

            #     # Nonlinearity testing
            #     leg_capacity_train_list, leg_capacity_test_list, leg_R2_train_list, leg_R2_test_list = nonlinearity_testing(input_data, output_data, leg_max_order, regressor, test_size, alpha, n_splits)
                
            #     # Memory testing
            #     mem_capacity_train_list, mem_capacity_test_list, mem_R2_train_list, mem_R2_test_list = memory_testing(input_data, output_data, max_timesteps_back, regressor, test_size, alpha, n_splits)

            #     onc_train = sum(leg_capacity_train_list)/len(leg_capacity_train_list)
            #     omc_train = sum(mem_capacity_train_list)/len(mem_capacity_train_list)

            #     onc_test = sum(leg_capacity_test_list)/len(leg_capacity_test_list)
            #     omc_test = sum(mem_capacity_test_list)/len(mem_capacity_test_list)
                
            # else:
            #     leg_capacity_train_list = np.nan
            #     mem_capacity_train_list = np.nan
            #     leg_capacity_test_list = np.nan
            #     mem_capacity_test_list = np.nan
            #     leg_R2_train_list = np.nan
            #     mem_R2_train_list = np.nan
            #     leg_R2_test_list = np.nan
            #     mem_R2_test_list = np.nan
            #     onc_train = np.nan
            #     omc_train = np.nan
            #     onc_test = onc_train
            #     omc_test = omc_train

            # Save results in dataframe
            df.at[j, 'num_threads'] = num_horizontal_threads
            df.at[j, 'alpha'] = alpha
            df.at[j, 'nonlinearity train'] = onc_train
            df.at[j, 'nonlinearity test'] = onc_test
            df.at[j, 'memory train'] = omc_train
            df.at[j, 'memory test'] = omc_test

            j = j + 1

            # npsavez_dict[f'{alpha}'] = (leg_capacity_train_list, leg_capacity_test_list, mem_capacity_train_list, mem_capacity_test_list)

        # np.savez(f"{path}{idx}_eval_{n_splits}_alphaCV.npz",
        #             npsavez_dict = npsavez_dict)
        
        print(idx, "eval done.")

    df.to_csv(f"{csv_name}.csv", index=False)