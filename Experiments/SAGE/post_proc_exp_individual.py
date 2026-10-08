import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn import preprocessing
from scipy.special import legendre
import matplotlib


if __name__ == "__main__":
    net_size = 6
    vid_path = os.path.join(os.path.dirname(__file__), f"{net_size}by{net_size}/Videos")

    sample_freq = 2
    eval_freq = 100
    angle = 25

    name = f"{sample_freq}_{eval_freq}_{angle}.MP4"

    data_path = f"{vid_path}/{name}_frames"

    test_size = 0.25
    alpha = 1
    clf = Ridge(alpha=alpha)
    
    leg_max_order = 10
    # t_max = 10
    rotate = False
    max_time_back_seconds = 1

    stimulation_point_label = 69 #34 #

    clean_data = pd.read_csv(f'{data_path}/clean_data.csv')
    clean_data['Time_s'] = clean_data['Time']/60
    min_label = clean_data['Label'].min()
    max_label = clean_data['Label'].max()

    if rotate:
        rotation = np.pi/2
        x = clean_data['X'].copy()
        y = clean_data['Y'].copy()
        clean_data['X'] = np.cos(rotation) * x - np.sin(rotation) * y
        clean_data['Y'] = np.sin(rotation) * x + np.cos(rotation) * y
        # x = clean_data['Xmm'].copy()
        # y = clean_data['Ymm'].copy()
        # clean_data['Xmm'] = np.cos(rotation) * x - np.sin(rotation) * y
        # clean_data['Ymm'] = np.sin(rotation) * x + np.cos(rotation) * y

    origin_x = (clean_data['X'].max()+clean_data['X'].min())/2 #clean_data['Xmm'].min()
    origin_y = (clean_data['Y'].max()+clean_data['Y'].min())/2 #clean_data['Ymm'].min()
    time_min = clean_data['Time_s'].min()
    time_max = clean_data['Time_s'].max()
    clean_data['Relative X'] = clean_data['X'] - origin_x
    clean_data['Relative Y'] = clean_data['Y'] - origin_y
    clean_data['Time_s'] = clean_data['Time_s'] - time_min

    num_labelled_frames_list = [len(clean_data[clean_data['Label'] == i]) for i in range(min_label, max_label+1)]

    min_shape = np.min(num_labelled_frames_list)
    print(min_shape, np.where(np.array(num_labelled_frames_list) == min_shape))

    clean_data.to_csv(f'{data_path}/clean_data_final.csv', index=False)

    '''

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    for j in range(min_label, max_label+1):
        label = j
        specific_label = clean_data[clean_data['Label'] == label]

        ax1.plot(specific_label['Relative X'], specific_label['Time_s'], label=label)
        ax2.plot(specific_label['Time_s'], specific_label['Relative Y'], label=label)
        ax3.plot(specific_label['Relative X'], specific_label['Relative Y'], label=label)

    ax1.set_xlabel("Relative X position ($px$)")
    ax1.set_ylabel("Time ($s$)")
    ax1.grid(True)
    # ax1.legend()
    ax1.set_title("Relative X positions over time")

    ax2.set_xlabel("Time ($s$)")
    ax2.set_ylabel("Y position ($px$)")
    ax2.grid(True)
    # ax2.invert_yaxis()
    # ax2.legend()
    ax2.set_title("Relative Y positions over time")

    ax3.set_xlabel("Relative X position ($mm$)")
    ax3.set_ylabel("Y position ($mm$)")
    ax3.grid(True)
    # ax3.invert_yaxis()
    # ax3.legend()
    ax3.set_title("Relative Trajectories of all points")

    plt.savefig(f"{data_path}/relative_trajectories.png")
    plt.close()

    stimulation_point = clean_data[clean_data['Label'] == stimulation_point_label]
    min_shape = np.min([len(clean_data[clean_data['Label'] == i]) for i in range(min_label, max_label+1)])
    print(min_shape)
    # max_time_back_seconds = t_max*62/min_shape
    t_max = np.rint(max_time_back_seconds * min_shape / 62).astype(int)

    for j in range(min_label, max_label+1):
        if j != stimulation_point_label:
            one_over_time = clean_data[clean_data['Label']==j]
            one_over_time = one_over_time.iloc[:, 5:] - one_over_time.iloc[0, 5:]
            array = one_over_time.to_numpy()
            array = array[:min_shape, :]
            if j == 0:
                output = array
            else:
                output = np.hstack([output, array])

    output = np.nan_to_num(output, nan=0)

    input_signal = stimulation_point['Relative Y'].to_numpy()
    input_signal = input_signal[:min_shape]
    input_signal = input_signal.reshape(-1, 1)
    input_signal = np.nan_to_num(input_signal, nan=0)
    plt.figure(figsize=(10, 5))
    plt.plot(input_signal, '-o', label='Input signal')
    plt.grid(True)
    plt.xlabel("Time ($s$)")
    plt.ylabel("Displacement of stimulation point (mm)")
    plt.title("Input signal over time")
    plt.savefig(f"{data_path}/input_signal.png")
    plt.close()
    
    # input_signal = -1 + (input_signal - np.min(input_signal)) * (1 - (-1)) / (np.max(input_signal) - np.min(input_signal))
    # scaler = preprocessing.StandardScaler()
    # input_signal = scaler.fit_transform(input_signal)
    input_signal = -1 + (input_signal - np.min(input_signal)) * (1 - (-1)) / (np.max(input_signal) - np.min(input_signal))

    plt.figure(figsize=(10, 5))
    plt.plot(input_signal, '-o', label='Input signal')
    plt.grid(True)
    plt.xlabel("Time ($s$)")
    plt.ylabel("Displacement of stimulation point (mm)")
    plt.title("Input signal over time")
    plt.savefig(f"{data_path}/input_signal_normalized.png")
    plt.close()

    # Nonlinearity capacity
    legendre_capacity_train = []
    legendre_capacity_test = []
    legendre_R2_train = []
    legendre_R2_test = []
    legendre_rmse_train = []
    legendre_rmse_test = []

    matplotlib.rc('pdf', fonttype=42)

    for n in range(1, leg_max_order+1):
        x = output
        polynomial = legendre(n)
        y = polynomial(input_signal)
        
        y2 = (1/len(y)) * np.sum((y - np.mean(y))**2)

        x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                            test_size=test_size,
                                                            random_state=42,
                                                            shuffle=True)
        
        train_size = y_train.shape[0]

        clf.fit(x_train, y_train)

        # Training capacity
        y_train_pred = clf.predict(x_train)

        idx = np.where(abs(y_train_pred) > 2)
        y_train_pred[idx] = np.mean(y)
        y_train[idx, 0] = np.mean(y)

        MSE_train = mean_squared_error(y_train, y_train_pred)
        capacity_train = 1 - MSE_train/y2
        R2_train = r2_score(y_train, y_train_pred)
        RMSE_train = root_mean_squared_error(y_train, y_train_pred)

        if R2_train < 0:
            R2_train = 0
        if capacity_train < 0:
            capacity_train = 0
        if RMSE_train < 0:
            RMSE_train = 0

        legendre_capacity_train.append(capacity_train)
        legendre_R2_train.append(R2_train)
        legendre_rmse_train.append(RMSE_train)

        # Testing capacity
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 2)
        y_test_pred[idx] = np.mean(y)
        y_test[idx, 0] = np.mean(y)

        MSE_test = mean_squared_error(y_test, y_test_pred)
        capacity_test = 1 - MSE_test/y2
        R2_test = r2_score(y_test, y_test_pred)
        RMSE_test = root_mean_squared_error(y_test, y_test_pred)

        if R2_test < 0:
            R2_test = 0
        if capacity_test < 0:
            capacity_test = 0
        if RMSE_test < 0:
            RMSE_test = 0

        legendre_capacity_test.append(capacity_test)
        legendre_R2_test.append(R2_test)
        legendre_rmse_test.append(RMSE_test)
        
        preds = np.concatenate((y_train_pred, y_test_pred), axis=0)

        # if n >=1 and n <= 5:
        #     print(f"Legendre Polynomial Order: {n}, Training Capacity: {capacity_train}, Testing Capacity: {capacity_test}")
        #     print(f"Legendre Polynomial Order: {n}, Training R2: {R2_train}, Testing R2: {R2_test}")
        #     print(f"Legendre Polynomial Order: {n}, Training RMSE: {RMSE_train}, Testing RMSE: {RMSE_test}")
        #     plt.figure(figsize=(7.5, 7.5))
        #     plt.plot(np.linspace(-1, 1, y.shape[0]), polynomial(np.linspace(-1, 1, y.shape[0])), linewidth=10, label='Target Value', zorder=0)
        #     plt.scatter(input_signal, preds, color='orange', s=10, label='Predicted Value', zorder=1)
        #     plt.xlabel("Input Signal")
        #     plt.ylabel("Desired Output")
        #     plt.title(f"Legendre Polynomial Order: {n}")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig(f"SMASIS_exp/SMASIS experiment results/6by6FishingLineNoPretension/legendre_plot_for_order_{n}.pdf", dpi=300)
        #     plt.close()
    
    # Plotting the results
    
    plt.figure(figsize=(7.5, 7.5))
    plt.plot(np.linspace(1, leg_max_order, leg_max_order), legendre_capacity_train, '-o', markersize = 10, linewidth = 2, label='Training cap')
    plt.plot(np.linspace(1, leg_max_order, leg_max_order), legendre_capacity_test, '-o', markersize = 10, linewidth = 2, label='Testing cap')
    plt.plot(np.linspace(1, leg_max_order, leg_max_order), legendre_R2_train, '-s', markersize = 10, linewidth = 2, label='Training R2')
    plt.plot(np.linspace(1, leg_max_order, leg_max_order), legendre_R2_test, '-s', markersize = 10, linewidth = 2, label='Testing R2')
    plt.xlabel("Legendre Polynomial Order")
    plt.ylabel("Capacity")
    plt.ylim(-0.1, 1.1)
    plt.title(f"Nonlinear Capacity {sum(legendre_capacity_train)/len(legendre_capacity_train):.4f}, {sum(legendre_capacity_test)/len(legendre_capacity_test):.4f}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{data_path}/nonlinear_capacity.png")
    plt.show()

    # Memory capacity
    memory_capacity_train = []
    memory_capacity_test = []
    memory_R2_train = []
    memory_R2_test = []
    memory_rmse_train = []
    memory_rmse_test = []

    # input_signal = input_signal[:3600, :]
    # output = output[:3600, :]

    time_data = stimulation_point['Time_s'].to_numpy()
    time_data = time_data[:min_shape]
    train_size = np.rint(min_shape * (1 - test_size)).astype(int)

    for t in range(t_max+1):
        x = output[t:]
        if t == 0:
            y = input_signal
        else:
            y = input_signal[:-t]

        y2 = (1/len(y)) * np.sum((y - np.mean(y))**2)

        x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                            test_size=test_size,
                                                            random_state=42,
                                                            shuffle=False)
            
        
        clf.fit(x_train, y_train)

        # Training capacity
        y_train_pred = clf.predict(x_train)

        idx = np.where(abs(y_train_pred) > 1.5)
        y_train_pred[idx] = np.mean(y)
        y_train[idx, 0] = np.mean(y)

        MSE_train = mean_squared_error(y_train, y_train_pred)
        capacity_train = 1 - MSE_train/y2
        R2_train = r2_score(y_train, y_train_pred)
        RMSE_train = root_mean_squared_error(y_train, y_train_pred)

        if R2_train < 0:
            R2_train = 0
        if capacity_train < 0:
            capacity_train = 0
        if RMSE_train < 0:
            RMSE_train = 0

        memory_capacity_train.append(capacity_train)
        memory_R2_train.append(R2_train)
        memory_rmse_train.append(RMSE_train)

        # Testing capacity
        y_test_pred = clf.predict(x_test)
        idx = np.where(abs(y_test_pred) > 1.5)
        y_test_pred[idx] = np.mean(y)
        y_test[idx, 0] = np.mean(y)

        MSE_test = mean_squared_error(y_test, y_test_pred)
        capacity_test = 1 - MSE_test/y2
        R2_test = r2_score(y_test, y_test_pred)
        RMSE_test = root_mean_squared_error(y_test, y_test_pred)

        if R2_test < 0:
            R2_test = 0
        if capacity_test < 0:
            capacity_test = 0
        if RMSE_test < 0:
            RMSE_test = 0

        memory_capacity_test.append(capacity_test)
        memory_R2_test.append(R2_test)
        memory_rmse_test.append(RMSE_test)

        # if t >= 0 and t <= 4:
        # if t == 0 or t == int(0.1*120) or t == int(0.2*120):
        #     print(f"Time: {t/120} seconds in the past, Training Capacity: {capacity_train}, Testing Capacity: {capacity_test}")
        #     print(f"Time: {t/120} seconds in the past, Training R2: {R2_train}, Testing R2: {R2_test}")
        #     print(f"Time: {t/120} seconds in the past, Training RMSE: {RMSE_train}, Testing RMSE: {RMSE_test}")
        #     plt.figure(figsize=(7.5, 7.5))
        #     plt.plot(np.linspace(0, len(y_test)/120, len(y_test)), input_signal[-1-len(y_test):-1], '-', linewidth = 4,  label=f'Actual Input at {t/120}')
        #     plt.plot(np.linspace(0, len(y_test)/120, len(y_test)), y_test, '-', markersize = 15, linewidth = 2.5,  label='Target Value')
        #     plt.plot(np.linspace(0, len(y_test)/120, len(y_test)), y_test_pred, '-', markersize = 10, linewidth = 1, label='Predicted Value')
        #     plt.xlabel("Time ($s$)")
        #     plt.ylabel("Desired Output")
        #     plt.title(f"Input from {t/120} seconds in the past")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig(f"SMASIS_exp/SMASIS experiment results/6by6FishingLineNoPretension/memory_plot_for_{t/120}sec_in_the_past.pdf", dpi=300)
        #     plt.close()
            
    # Plotting the results
    plt.figure(figsize=(7.5, 7.5))
    plt.plot(np.linspace(0, max_time_back_seconds, t_max+1), memory_capacity_train, '-o', markersize = 10, linewidth = 2, label='Training capacity')
    plt.plot(np.linspace(0, max_time_back_seconds, t_max+1), memory_capacity_test, '-o', markersize = 10, linewidth = 2, label='Testing capacity')
    plt.plot(np.linspace(0, max_time_back_seconds, t_max+1), memory_R2_train, '-s', markersize = 10, linewidth = 2, label='Training R2')
    plt.plot(np.linspace(0, max_time_back_seconds, t_max+1), memory_R2_test, '-s', markersize = 10, linewidth = 2, label='Testing R2')
    plt.xlabel("Seconds in the past")
    plt.ylabel("Capacity")
    plt.ylim(-0.1, 1.1)
    plt.title(f"Memory Capacity {sum(memory_capacity_train)/len(memory_capacity_train):.4f}, {sum(memory_capacity_test)/len(memory_capacity_test):.4f}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{data_path}/memory_capacity.png")
    plt.show()

    print(f"Overall Nonlinearity Capacity: {sum(legendre_capacity_test)/len(legendre_capacity_test)}, Overall Memory Capacity: {sum(memory_capacity_test)/len(memory_capacity_test)}")

    # np.savez(f"./Experiments/SMASIS/SMASIS_exp/SMASIS experiment results/2by2FishingLineNoPretension/legendre_memory_testing_revised.npz", 
    #          legendre_capacity_test=legendre_capacity_test, 
    #          memory_capacity_test=memory_capacity_test,
    #          legendre_R2_test=legendre_R2_test,
    #          memory_R2_test=memory_R2_test,
    #          legendre_rmse_test=legendre_rmse_test,
    #          memory_rmse_test=memory_rmse_test)

    # leg_max_order = 10
    # t_max = 10
    # t_heatmap = np.linspace(0, max_time_back_seconds, t_max+1)

    # capacity_train_matrix = np.zeros((leg_max_order, t_max+1))
    # capacity_test_matrix = np.zeros((leg_max_order, t_max+1))
    # R2_train_matrix = np.zeros((leg_max_order, t_max+1))
    # R2_test_matrix = np.zeros((leg_max_order, t_max+1))

    # for n in range(1, leg_max_order+1):
    #     leg = legendre(n)
    #     for t_i in range(t_max+1):
    #         t = int(t_heatmap[t_i] * 120)
    #         x = output[t:]
    #         if t == 0:
    #             y = leg(input_signal)
    #         else:
    #             y = leg(input_signal[:-t])

    #         y2 = (1/len(y)) * np.sum((y - np.mean(y))**2)
        
    #         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
    #         clf.fit(x_train, y_train)

    #         # Training capacity
    #         y_train_pred = clf.predict(x_train)

    #         MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
    #         capacity_train = 1 - MSE/y2
    #         R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

    #         # Testing capacity
    #         y_test_pred = clf.predict(x_test)
            
    #         idx = np.where(abs(y_test_pred) > 2)
    #         y_test_pred[idx] = np.mean(y)
    #         y_test[idx, 0] = np.mean(y)

    #         MSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    #         capacity_test = 1 - MSE/y2
    #         R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

    #         if R2_train < 0:
    #             R2_train = 0
    #         if R2_test < 0:
    #             R2_test = 0
    #         if capacity_train < 0:
    #             capacity_train = 0
    #         if capacity_test < 0:
    #             capacity_test = 0

    #         capacity_train_matrix[n-1, t_i] = capacity_train
    #         capacity_test_matrix[n-1, t_i] = capacity_test
    #         R2_train_matrix[n-1, t_i] = R2_train
    #         R2_test_matrix[n-1, t_i] = R2_test

    # Plotting the results
    # fig = plt.figure(figsize=(7.5, 7.5))
    # ax = plt.axes()
    # img = ax.matshow(capacity_test_matrix, cmap=matplotlib.colormaps['RdYlBu_r'], vmin=0, vmax=1)
    # ax.set_title("Variation of Nonlinear and Memory Capacity")
    # ax.set_ylabel("Legendre Polynomial Order")
    # ax.set_xlabel("Seconds in the Past")
    # ax.set_yticks(np.arange(leg_max_order))
    # ax.set_yticklabels(np.arange(1, leg_max_order + 1))
    # ax.set_xticks(np.arange(t_max+1))
    # ax.set_xticklabels(np.round(t_heatmap, 2))
    # ax.xaxis.set_ticks_position('bottom')  # Move xticks to the bottom
    # ax.xaxis.set_label_position('bottom')  # Move xlabel to the bottom
    # cbar = plt.colorbar(img, shrink = 0.6)
    # # cbar.set_label("Capacity", rotation=270, labelpad=15)
    # plt.savefig(f"SMASIS_exp/SMASIS experiment results/6b64FishingLineNoPretension/heatmap_testing.pdf", dpi=300)
    # plt.close()

    '''