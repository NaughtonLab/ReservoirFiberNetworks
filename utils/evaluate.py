import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.special import legendre

'''
Single train/test split evaluation of the reservoir (nonlinearity, memory and the combined
nonlinearity-memory matrix). Cross-validated versions of nonlinearity_testing and memory_testing
live in utils/evaluate_cv.py and are used when CV=True is passed.
'''

def get_regressor(regressor, alpha):
    if regressor == "Lin":
        ### Linear Regression
        return LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        return Ridge(alpha=alpha)
    raise ValueError(f"Unknown regressor '{regressor}'. Use 'Lin' or 'Rid'.")

def __fit__(x_train, x_test, y_train, y_test, clf, y, return_predictions=False):
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

    if return_predictions:
        return capacity_train, capacity_test, R2_train, R2_test, y_test, y_test_pred
    return capacity_train, capacity_test, R2_train, R2_test

def save_predictions(file_path, y_test_list, y_test_pred_list, labels):
    '''
    Saves the test targets and predictions of every Legendre order / delay in one npz:
    keys y_test_<label> and y_test_pred_<label> (e.g. y_test_leg3, y_test_pred_delay10), both 1-D.
    y_test is the target after the |pred|>1 clipping in __fit__, i.e. what the scores were computed on.
    '''
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    arrays = {}
    for label, y_test, y_test_pred in zip(labels, y_test_list, y_test_pred_list):
        arrays[f'y_test_{label}'] = np.ravel(y_test)
        arrays[f'y_test_pred_{label}'] = np.ravel(y_test_pred)
    np.savez(file_path, **arrays)
    print("Predictions saved in", file_path)

def nonlinearity_testing(input, output, leg_max_order, regressor, test_size, alpha, *, CV=False, type_CV=None, n_splits=None, save_predictions_path=None):
    if CV:
        from utils.evaluate_cv import nonlinearity_testing_cv
        if save_predictions_path is not None:
            print("WARNING: Saving predictions is not supported with cross validation (CV).")
        return nonlinearity_testing_cv(input, output, leg_max_order, regressor, alpha, type_CV, n_splits)

    clf = get_regressor(regressor, alpha)
    keep = save_predictions_path is not None

    x = output
    capacity_train_list = []
    capacity_test_list = []
    R2_train_list = []
    R2_test_list = []
    y_test_list = []
    y_test_pred_list = []
    for n in range(1, leg_max_order+1):
        leg = legendre(n)
        y = leg(input)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

        fit = __fit__(x_train, x_test, y_train, y_test, clf, y, return_predictions=keep)
        capacity_train, capacity_test, R2_train, R2_test = fit[:4]
        if keep:
            y_test_list.append(fit[4])
            y_test_pred_list.append(fit[5])

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

    if keep:
        save_predictions(save_predictions_path, y_test_list, y_test_pred_list, [f'leg{n}' for n in range(1, leg_max_order+1)])

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def memory_testing(input, output, max_timesteps_back, regressor, test_size, alpha, *, CV=False, type_CV=None, n_splits=None, save_predictions_path=None):
    if CV:
        from utils.evaluate_cv import memory_testing_cv
        if save_predictions_path is not None:
            print("WARNING: Saving predictions is not supported with cross validation (CV).")
        return memory_testing_cv(input, output, max_timesteps_back, regressor, alpha, type_CV, n_splits)

    clf = get_regressor(regressor, alpha)
    keep = save_predictions_path is not None

    capacity_train_list = []
    capacity_test_list = []
    R2_train_list = []
    R2_test_list = []
    y_test_list = []
    y_test_pred_list = []
    for n in range(0, max_timesteps_back+1):
        x = output[n:]
        if n == 0:
            y = input
        else:
            y = input[:-n]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

        fit = __fit__(x_train, x_test, y_train, y_test, clf, y, return_predictions=keep)
        capacity_train, capacity_test, R2_train, R2_test = fit[:4]
        if keep:
            y_test_list.append(fit[4])
            y_test_pred_list.append(fit[5])

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

    if keep:
        save_predictions(save_predictions_path, y_test_list, y_test_pred_list, [f'delay{n}' for n in range(0, max_timesteps_back+1)])

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def nonlinearity_memory_matrix(input, output, leg_max_order, max_timesteps_back, regressor, test_size, alpha):
    clf = get_regressor(regressor, alpha)

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

            capacity_train, capacity_test, R2_train, R2_test = __fit__(x_train, x_test, y_train, y_test, clf, y)

            capacity_train_matrix[n-1, t] = capacity_train
            capacity_test_matrix[n-1, t] = capacity_test
            R2_train_matrix[n-1, t] = R2_train
            R2_test_matrix[n-1, t] = R2_test

    return capacity_train_matrix, capacity_test_matrix, R2_train_matrix, R2_test_matrix
