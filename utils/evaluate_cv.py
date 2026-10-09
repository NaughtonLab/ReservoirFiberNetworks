import numpy as np
from scipy.special import legendre
from sklearn.model_selection import KFold, TimeSeriesSplit

from utils.evaluate import get_regressor, __fit__

'''
Cross-validated versions of nonlinearity_testing and memory_testing. They are called from
utils/evaluate.py when CV=True. Each metric is the mean over the n_splits folds.
Supported type_CV: 'KFold' (contiguous folds, no shuffling) and 'TimeSeriesSplit'.
'''

def get_splitter(type_CV, n_splits):
    match type_CV:
        case "KFold":
            return KFold(n_splits=n_splits, shuffle=False)
        case "TimeSeriesSplit":
            return TimeSeriesSplit(n_splits=n_splits)
        case _:
            raise ValueError("Please specify the type of cross validation (CV) to perform. Supported types are 'KFold' and 'TimeSeriesSplit'.")

def __cv_fit__(x, y, clf, splitter, n_splits):
    cv_capacity_train_sum = 0
    cv_capacity_test_sum = 0
    cv_R2_train_sum = 0
    cv_R2_test_sum = 0
    for train_idx, test_idx in splitter.split(x):
        x_train, x_test, y_train, y_test = x[train_idx, :], x[test_idx, :], y[train_idx, :], y[test_idx, :]

        capacity_train, capacity_test, R2_train, R2_test = __fit__(x_train, x_test, y_train, y_test, clf, y)
        cv_capacity_train_sum += capacity_train
        cv_capacity_test_sum += capacity_test
        cv_R2_train_sum += R2_train
        cv_R2_test_sum += R2_test

    return cv_capacity_train_sum/n_splits, cv_capacity_test_sum/n_splits, cv_R2_train_sum/n_splits, cv_R2_test_sum/n_splits

def nonlinearity_testing_cv(input, output, leg_max_order, regressor, alpha, type_CV, n_splits):
    clf = get_regressor(regressor, alpha)
    splitter = get_splitter(type_CV, n_splits)

    capacity_train_list = []
    capacity_test_list = []
    R2_train_list = []
    R2_test_list = []
    for n in range(1, leg_max_order+1):
        y = legendre(n)(input)
        capacity_train, capacity_test, R2_train, R2_test = __cv_fit__(output, y, clf, splitter, n_splits)

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def memory_testing_cv(input, output, max_timesteps_back, regressor, alpha, type_CV, n_splits):
    clf = get_regressor(regressor, alpha)
    splitter = get_splitter(type_CV, n_splits)

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

        capacity_train, capacity_test, R2_train, R2_test = __cv_fit__(x, y, clf, splitter, n_splits)

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list
