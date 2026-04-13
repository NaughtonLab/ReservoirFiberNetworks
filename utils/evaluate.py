import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.special import legendre
import matplotlib.pyplot as plt

def __fit__(x_train, x_test, y_train, y_test, clf, y, plot=False):
    y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)

    # Training
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    
    idx = np.where(abs(y_train_pred) > 1)
    y_train_pred[idx] = np.mean(y)
    y_train[idx, 0] = np.mean(y)
    
    y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)

    MSE_train = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
    capacity_train = 1 - MSE_train/y2_train
    R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

    # Testing
    y_test_pred = clf.predict(x_test)

    idx = np.where(abs(y_test_pred) > 1)
    y_test_pred[idx] = np.mean(y)
    y_test[idx, 0] = np.mean(y)

    y2_test = (1/len(y_test)) * np.sum((y_test-np.mean(y_test))**2)

    MSE_test = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    capacity_test = 1 - MSE_test/y2_test
    R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

    if R2_test < 0:
        R2_test = 0
    if R2_train < 0:
        R2_train = 0
    if capacity_test < 0:
        capacity_test = 0
    if capacity_train < 0:
        capacity_train = 0

    if plot:
        print("MSE", MSE_train, MSE_test, "y2", y2, "capacity", capacity_train, capacity_test)
        return capacity_train, capacity_test, R2_train, R2_test, y_train, y_train_pred, y_test, y_test_pred
    else:
        return capacity_train, capacity_test, R2_train, R2_test

def nonlinearity_testing(input, output, leg_max_order, regressor, test_size, alpha, plot=False, CV=False, **kwargs):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    if CV:
        if plot:
            print("WARNING: Plotting Predictions is not supported when performing cross validation (CV). This will be implemented in the future.")
        type_CV = kwargs.get('type_CV', None)
        match type_CV:            
            case "KFold":
                from sklearn.model_selection import KFold
                n_splits = kwargs.get('n_splits', None)
                kf = KFold(n_splits=n_splits, shuffle=False)
            case "TimeSeriesSplit":
                from sklearn.model_selection import TimeSeriesSplit
                n_splits = kwargs.get('n_splits', None)
                tscv = TimeSeriesSplit(n_splits=n_splits)
            case _:
                raise ValueError("Please specify the type of cross validation (CV) to perform. Supported types are 'KFold' and 'TimeSeriesSplit'.")

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

        if CV:
            cv_capacity_train_sum = 0
            cv_capacity_test_sum = 0
            cv_R2_train_sum = 0
            cv_R2_test_sum = 0
            match type_CV:
                case "KFold":
                    split = kf.split(x)
                case "TimeSeriesSplit":
                    split = tscv.split(x)
            for i, (train_idx, test_idx) in enumerate(split):
                # print(f"Fold {i}")
                x_train, x_test, y_train, y_test = x[train_idx, :], x[test_idx, :], y[train_idx, :], y[test_idx, :]

                capacity_train, capacity_test, R2_train, R2_test = __fit__(x_train, x_test, y_train, y_test, clf, y, plot=False)
                cv_capacity_train_sum += capacity_train
                cv_capacity_test_sum += capacity_test
                cv_R2_train_sum += R2_train
                cv_R2_test_sum += R2_test

            capacity_train = cv_capacity_train_sum / n_splits
            capacity_test = cv_capacity_test_sum / n_splits
            R2_train = cv_R2_train_sum / n_splits
            R2_test = cv_R2_test_sum / n_splits             

        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

            if plot:
                print("Legendre", n)
                capacity_train, capacity_test, R2_train, R2_test, y_train, y_train_pred, y_test, y_test_pred = __fit__(x_train, x_test, y_train, y_test, clf, y, plot=True)

                plt.figure(figsize=(15, 5))
                plt.subplot(121)
                plt.scatter(input[:train_size, :], y_train, label='True')
                plt.scatter(input[:train_size, :],y_train_pred, label='Predicted')
                plt.title(f'Training Legendre {n}')
                plt.legend()
                plt.grid()

                plt.subplot(122)
                plt.scatter(input[train_size:, :], y_test, label='True')
                plt.scatter(input[train_size:, :], y_test_pred, label='Predicted')
                plt.title(f'Testing Legendre {n}')
                plt.legend()
                plt.grid()

                plt.show()  
            else:
                capacity_train, capacity_test, R2_train, R2_test = __fit__(x_train, x_test, y_train, y_test, clf, y, plot=False)   

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def memory_testing(input, output, max_timesteps_back, regressor, test_size, alpha, plot=False, CV=False, **kwargs):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    if CV:
        if plot:
            print("WARNING: Plotting Predictions is not supported when performing cross validation (CV). This will be implemented in the future.")
        type_CV = kwargs.get('type_CV', None)
        match type_CV:            
            case "KFold":
                from sklearn.model_selection import KFold
                n_splits = kwargs.get('n_splits', None)
                kf = KFold(n_splits=n_splits, shuffle=False)
            case "TimeSeriesSplit":
                from sklearn.model_selection import TimeSeriesSplit
                n_splits = kwargs.get('n_splits', None)
                tscv = TimeSeriesSplit(n_splits=n_splits)
            case _:
                raise ValueError("Please specify the type of cross validation (CV) to perform. Supported types are 'KFold' and 'TimeSeriesSplit'.")

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

        if CV:
            cv_capacity_train_sum = 0
            cv_capacity_test_sum = 0
            cv_R2_train_sum = 0
            cv_R2_test_sum = 0
            match type_CV:
                case "KFold":
                    split = kf.split(x)
                case "TimeSeriesSplit":
                    split = tscv.split(x)
            for i, (train_idx, test_idx) in enumerate(split):
                # print(f"Fold {i}")
                x_train, x_test, y_train, y_test = x[train_idx, :], x[test_idx, :], y[train_idx, :], y[test_idx, :]

                capacity_train, capacity_test, R2_train, R2_test = __fit__(x_train, x_test, y_train, y_test, clf, y, plot=False)
                cv_capacity_train_sum += capacity_train
                cv_capacity_test_sum += capacity_test
                cv_R2_train_sum += R2_train
                cv_R2_test_sum += R2_test

            capacity_train = cv_capacity_train_sum / n_splits
            capacity_test = cv_capacity_test_sum / n_splits
            R2_train = cv_R2_train_sum / n_splits
            R2_test = cv_R2_test_sum / n_splits

        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

            if plot:
                print("Memory", n)
                capacity_train, capacity_test, R2_train, R2_test, y_train, y_train_pred, y_test, y_test_pred = __fit__(x_train, x_test, y_train, y_test, clf, y, plot=True)

                plt.figure(figsize=(15, 5))
                plt.subplot(121)
                plt.scatter(y_train, y_train, label='True')
                plt.plot(y_train_pred, label='Predicted')
                plt.title(f'Training Memory {n}')
                plt.legend()
                plt.grid()

                plt.subplot(122)
                plt.plot(y_test, label='True')
                plt.plot(y_test_pred, label='Predicted')
                plt.title(f'Testing Memory {n}')
                plt.legend()
                plt.grid()

                plt.show()
            else:
                capacity_train, capacity_test, R2_train, R2_test = __fit__(x_train, x_test, y_train, y_test, clf, y, plot=False)

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

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
        
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

            capacity_train, capacity_test, R2_train, R2_test = __fit__(x_train, x_test, y_train, y_test, clf, y, plot=False)

            capacity_train_matrix[n-1, t] = capacity_train
            capacity_test_matrix[n-1, t] = capacity_test
            R2_train_matrix[n-1, t] = R2_train
            R2_test_matrix[n-1, t] = R2_test

    return capacity_train_matrix, capacity_test_matrix, R2_train_matrix, R2_test_matrix