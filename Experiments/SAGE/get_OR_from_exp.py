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

def nonlinearity_testing(input_signal, output, leg_max_order, test_size, alpha, regressor):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    legendre_capacity_train = []
    legendre_capacity_test = []
    legendre_R2_train = []
    legendre_R2_test = []
    legendre_rmse_train = []
    legendre_rmse_test = []

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

        y2_train = (1/len(y_train)) * np.sum((y_train - np.mean(y_train))**2)

        MSE_train = mean_squared_error(y_train, y_train_pred)
        capacity_train = 1 - MSE_train/y2_train
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

        y2_test = (1/len(y_test)) * np.sum((y_test - np.mean(y_test))**2)

        MSE_test = mean_squared_error(y_test, y_test_pred)
        capacity_test = 1 - MSE_test/y2_test
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

        return legendre_capacity_train, legendre_capacity_test, legendre_R2_train, legendre_R2_test, legendre_rmse_train, legendre_rmse_test

def memory_testing(input_signal, output, max_time_back, test_size, alpha, regressor):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    memory_capacity_train = []
    memory_capacity_test = []
    memory_R2_train = []
    memory_R2_test = []
    memory_rmse_train = []
    memory_rmse_test = []

    for t in range(max_time_back+1):
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

        y2_train = (1/len(y_train)) * np.sum((y_train - np.mean(y_train))**2)

        MSE_train = mean_squared_error(y_train, y_train_pred)
        capacity_train = 1 - MSE_train/y2_train
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

        y2_test = (1/len(y_test)) * np.sum((y_test - np.mean(y_test))**2)

        MSE_test = mean_squared_error(y_test, y_test_pred)
        capacity_test = 1 - MSE_test/y2_test
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

    return memory_capacity_train, memory_capacity_test, memory_R2_train, memory_R2_test, memory_rmse_train, memory_rmse_test

if __name__ == "__main__":
    net_size = 4 #6 #
    vid_path = os.path.join(os.path.dirname(__file__), f"{net_size}by{net_size}/Videos")
    save_path = "C:/D/ABHYAAS/OneDrive - Virginia Tech/naughtonlab - active_projects/2025-JIMSS_paper/pdf_v3/Fig 4"

    sample_freq = 2
    eval_freq = 100
    angle_list = [25, 30, 35, 40, 45, 50]

    test_size = 0.25
    
    leg_max_order = 10
    # max_timesteps_back = 10
    rotate = False
    max_time_back_seconds = 1
    fps = 120
    frame_rate_for_img = 1

    alpha_leg = 1
    alpha_mem = 10

    stimulation_point_label = 34 #69 #

    df = pd.read_csv(f"{vid_path}/alpha_leg {alpha_leg} alpha_mem {alpha_mem}/grid_search_results_{sample_freq}_{eval_freq}.csv")
    print("Hors Y + Vers X")

    for i in range(len(angle_list)):
        angle = angle_list[i]

        name = f"{sample_freq}_{eval_freq}_{angle}.MP4"

        data_path = f"{vid_path}/{name}_frames"

        hors_xy = pd.read_csv(f'{data_path}/hors_xy.csv')
        label_list_1 = hors_xy['Label'].unique().tolist()
        min_label_1 = min(label_list_1)
        max_label_1 = max(label_list_1)
        min_shape_1 = np.min([len(hors_xy[hors_xy['Label'] == i]) for i in label_list_1])
        vers_xy = pd.read_csv(f'{data_path}/vers_xy.csv')
        label_list_2 = vers_xy['Label'].unique().tolist()
        min_label_2 = min(label_list_2)
        max_label_2 = max(label_list_2)
        min_shape_2 = np.min([len(vers_xy[vers_xy['Label'] == i]) for i in label_list_2])

        clean_data = pd.read_csv(f'{data_path}/clean_data_final.csv')
        clean_data['Time_s'] = clean_data['Time']/60
        min_label = clean_data['Label'].min()
        max_label = clean_data['Label'].max()

        stimulation_point = clean_data[clean_data['Label'] == stimulation_point_label]
        # min_shape = np.min([len(clean_data[clean_data['Label'] == i]) for i in range(min_label, max_label+1)])
        min_shape = np.min([min_shape_1, min_shape_2])
        print(min_shape)

        output = None
        for j in range(min_label_1, max_label_1+1):
            if j != stimulation_point_label:
                one_over_time = hors_xy[hors_xy['Label']==j]
                if one_over_time.empty:
                    continue
                one_over_time = one_over_time.iloc[:, 6:7] - one_over_time.iloc[0, 6:7]
                array = one_over_time.to_numpy()
                array = array[:min_shape, :]
                if output is None:
                    output = array
                else:
                    output = np.hstack([output, array])
        
        for j in range(min_label_2, max_label_2+1):
            if j != stimulation_point_label:
                one_over_time = vers_xy[vers_xy['Label']==j]
                if one_over_time.empty:
                    continue
                one_over_time = one_over_time.iloc[:, 5:6] - one_over_time.iloc[0, 5:6]
                array = one_over_time.to_numpy()
                array = array[:min_shape, :]
                if output is None:
                    output = array
                else:
                    output = np.hstack([output, array])
        
        output /= np.mean(output, axis=0)
        output /= np.std(output, axis=0)
        scaler = preprocessing.StandardScaler()
        output = scaler.fit_transform(output)

        input_signal = stimulation_point['Relative Y'].to_numpy()
        input_signal = input_signal[:min_shape]
        input_signal = input_signal.reshape(-1, 1)
        input_signal = np.nan_to_num(input_signal, nan=0)

        input_signal = -1 + (input_signal - np.min(input_signal)) * (1 - (-1)) / (np.max(input_signal) - np.min(input_signal))

        if i == 1 and eval_freq == 2 and net_size == 6:
            transience = 0
        else:
            transience = 500

        input_signal = input_signal[transience:, :]
        output = output[transience:, :]
        print(output.shape, input_signal.shape)

        # Nonlinearity testing
        legendre_capacity_train, legendre_capacity_test, legendre_R2_train, legendre_R2_test, legendre_rmse_train, legendre_rmse_test = nonlinearity_testing(input_signal, output, leg_max_order, test_size, alpha_leg)

        # Memory testing
        time_data = stimulation_point['Time_s'].to_numpy()
        time_data = time_data[:min_shape]
        train_size = np.rint(min_shape * (1 - test_size)).astype(int)
        max_timesteps_back = np.rint(max_time_back_seconds * fps/frame_rate_for_img).astype(int)
        memory_capacity_train, memory_capacity_test, memory_R2_train, memory_R2_test, memory_rmse_train, memory_rmse_test = memory_testing(input_signal, output, max_timesteps_back, test_size, alpha_mem)

        np.savez(f"{data_path}/hors_y_vers_x_results.npz",
                 input_data=input_signal,
                 output_data=output,
                 legendre_capacity_test=legendre_capacity_test,
                 legendre_R2_test=legendre_R2_test,
                 legendre_rmse_test=legendre_rmse_test,
                 memory_capacity_test=memory_capacity_test,
                 memory_R2_test=memory_R2_test,
                 memory_rmse_test=memory_rmse_test)

        df.at[i, 'onc_train_hyvx'] = sum(legendre_capacity_train)/len(legendre_capacity_train)
        df.at[i, 'omc_train_hyvx'] = sum(memory_capacity_train)/len(memory_capacity_train)
        df.at[i, 'onc_test_hyvx'] = sum(legendre_capacity_test)/len(legendre_capacity_test)
        df.at[i, 'omc_test_hyvx'] = sum(memory_capacity_test)/len(memory_capacity_test)

    print("Near Springs")

    for i in range(len(angle_list)):
        angle = angle_list[i]

        name = f"{sample_freq}_{eval_freq}_{angle}.MP4"

        data_path = f"{vid_path}/{name}_frames"

        near_springs = pd.read_csv(f'{data_path}/near_springs.csv')
        label_list_ns = near_springs['Label'].unique().tolist()
        min_label_ns = min(label_list_ns)
        max_label_ns = max(label_list_ns)
        min_shape_ns = np.min([len(near_springs[near_springs['Label'] == i]) for i in label_list_ns])

        clean_data = pd.read_csv(f'{data_path}/clean_data_final.csv')
        clean_data['Time_s'] = clean_data['Time']/60
        min_label = clean_data['Label'].min()
        max_label = clean_data['Label'].max()

        stimulation_point = clean_data[clean_data['Label'] == stimulation_point_label]
        min_shape = np.min([len(clean_data[clean_data['Label'] == i]) for i in range(min_label, max_label+1)])
        min_shape = np.min([min_shape_ns, min_shape])
        print(min_shape)
        max_time_back = np.rint(max_time_back_seconds * min_shape / 62).astype(int)

        output = None
        for j in range(min_label_ns, max_label_ns+1):
            if j != stimulation_point_label:
                one_over_time = near_springs[near_springs['Label']==j]
                if one_over_time.empty:
                    continue
                one_over_time = one_over_time.iloc[:, 5:] - one_over_time.iloc[0, 5:]
                array = one_over_time.to_numpy()
                array = array[:min_shape, :]
                if output is None:
                    output = array
                else:
                    output = np.hstack([output, array])

        output /= np.mean(output, axis=0)
        output /= np.std(output, axis=0)
        scaler = preprocessing.StandardScaler()
        output = scaler.fit_transform(output)

        input_signal = stimulation_point['Relative Y'].to_numpy()
        input_signal = input_signal[:min_shape]
        input_signal = input_signal.reshape(-1, 1)
        input_signal = np.nan_to_num(input_signal, nan=0)

        input_signal = -1 + (input_signal - np.min(input_signal)) * (1 - (-1)) / (np.max(input_signal) - np.min(input_signal))

        if i == 1 and eval_freq == 2 and net_size == 6:
            transience = 0
        else:
            transience = 500

        input_signal = input_signal[transience:, :]
        output = output[transience:, :]
        print(output.shape, input_signal.shape)

        # Nonlinearity testing
        legendre_capacity_train, legendre_capacity_test, legendre_R2_train, legendre_R2_test, legendre_rmse_train, legendre_rmse_test = nonlinearity_testing(input_signal, output, leg_max_order, test_size, alpha_leg)

        # Memory testing
        time_data = stimulation_point['Time_s'].to_numpy()
        time_data = time_data[:min_shape]
        train_size = np.rint(min_shape * (1 - test_size)).astype(int)
        max_timesteps_back = np.rint(max_time_back_seconds * fps/frame_rate_for_img).astype(int)
        memory_capacity_train, memory_capacity_test, memory_R2_train, memory_R2_test, memory_rmse_train, memory_rmse_test = memory_testing(input_signal, output, max_timesteps_back, test_size, alpha_mem)
        
        np.savez(f"{data_path}/near_springs_results.npz",
                 input_data=input_signal,
                 output_data=output,
                 legendre_capacity_test=legendre_capacity_test,
                 legendre_R2_test=legendre_R2_test,
                 legendre_rmse_test=legendre_rmse_test,
                 memory_capacity_test=memory_capacity_test,
                 memory_R2_test=memory_R2_test,
                 memory_rmse_test=memory_rmse_test)

        df.at[i, 'onc_train_ns'] = sum(legendre_capacity_train)/len(legendre_capacity_train)
        df.at[i, 'omc_train_ns'] = sum(memory_capacity_train)/len(memory_capacity_train)
        df.at[i, 'onc_test_ns'] = sum(legendre_capacity_test)/len(legendre_capacity_test)
        df.at[i, 'omc_test_ns'] = sum(memory_capacity_test)/len(memory_capacity_test)

    print("Near Actuation")

    for i in range(len(angle_list)):
        angle = angle_list[i]

        name = f"{sample_freq}_{eval_freq}_{angle}.MP4"

        data_path = f"{vid_path}/{name}_frames"

        near_act = pd.read_csv(f'{data_path}/near_actuation.csv')
        label_list_na = near_act['Label'].unique().tolist()
        min_label_na = min(label_list_na)
        max_label_na = max(label_list_na)
        min_shape_na = np.min([len(near_act[near_act['Label'] == i]) for i in label_list_na])
        clean_data = pd.read_csv(f'{data_path}/clean_data_final.csv')
        clean_data['Time_s'] = clean_data['Time']/60
        min_label = clean_data['Label'].min()
        max_label = clean_data['Label'].max()

        stimulation_point = clean_data[clean_data['Label'] == stimulation_point_label]
        min_shape = np.min([len(clean_data[clean_data['Label'] == i]) for i in range(min_label, max_label+1)])
        min_shape = np.min([min_shape_ns, min_shape])
        print(min_shape)

        output = None
        for j in range(min_label_na, max_label_na+1):
            if j != stimulation_point_label:
                one_over_time = near_act[near_act['Label']==j]
                if one_over_time.empty:
                    continue
                one_over_time = one_over_time.iloc[:, 5:] - one_over_time.iloc[0, 5:]
                array = one_over_time.to_numpy()
                array = array[:min_shape, :]
                if output is None:
                    output = array
                else:
                    output = np.hstack([output, array])

        output = np.nan_to_num(output, nan=0)
        output /= np.mean(output, axis=0)
        output /= np.std(output, axis=0)
        scaler = preprocessing.StandardScaler()
        output = scaler.fit_transform(output)

        input_signal = stimulation_point['Relative Y'].to_numpy()
        input_signal = input_signal[:min_shape]
        input_signal = input_signal.reshape(-1, 1)
        input_signal = np.nan_to_num(input_signal, nan=0)

        input_signal = -1 + (input_signal - np.min(input_signal)) * (1 - (-1)) / (np.max(input_signal) - np.min(input_signal))

        if i == 1 and eval_freq == 2 and net_size == 6:
            transience = 0
        else:
            transience = 500

        input_signal = input_signal[transience:, :]
        output = output[transience:, :]
        print(output.shape, input_signal.shape)

        # Nonlinearity testing
        legendre_capacity_train, legendre_capacity_test, legendre_R2_train, legendre_R2_test, legendre_rmse_train, legendre_rmse_test = nonlinearity_testing(input_signal, output, leg_max_order, test_size, alpha_leg)

        # Memory testing
        time_data = stimulation_point['Time_s'].to_numpy()
        time_data = time_data[:min_shape]
        train_size = np.rint(min_shape * (1 - test_size)).astype(int)
        max_timesteps_back = np.rint(max_time_back_seconds * fps/frame_rate_for_img).astype(int)
        memory_capacity_train, memory_capacity_test, memory_R2_train, memory_R2_test, memory_rmse_train, memory_rmse_test = memory_testing(input_signal, output, max_timesteps_back, test_size, alpha_mem)

        np.savez(f"{data_path}/near_actuation_results.npz",
                 input_data=input_signal,
                 output_data=output,
                 legendre_capacity_test=legendre_capacity_test,
                 legendre_R2_test=legendre_R2_test,
                 legendre_rmse_test=legendre_rmse_test,
                 memory_capacity_test=memory_capacity_test,
                 memory_R2_test=memory_R2_test,
                 memory_rmse_test=memory_rmse_test)

        df.at[i, 'onc_train_na'] = sum(legendre_capacity_train)/len(legendre_capacity_train)
        df.at[i, 'omc_train_na'] = sum(memory_capacity_train)/len(memory_capacity_train)
        df.at[i, 'onc_test_na'] = sum(legendre_capacity_test)/len(legendre_capacity_test)
        df.at[i, 'omc_test_na'] = sum(memory_capacity_test)/len(memory_capacity_test)

    df.to_csv(f"{vid_path}/grid_search_results_with_OR_{sample_freq}_{eval_freq}.csv", index=False)

    plt.figure(figsize=(7.5, 7.5))
    plt.plot(angle_list, df['onc_test'], '-o', markersize = 10, linewidth = 3, label='All Outputs')
    plt.plot(angle_list, df['onc_test_hyvx'], '-o', markersize = 10, linewidth = 2, label='Horizontal Y + Vertical X')
    plt.plot(angle_list, df['onc_test_ns'], '-o', markersize = 10, linewidth = 2, label='Near Springs')
    plt.plot(angle_list, df['onc_test_na'], '-o', markersize = 10, linewidth = 2, label='Near Actuation')
    plt.xlabel("Angle of Stimulation (deg)")
    plt.ylabel("Capacity")
    plt.ylim(-0.1, 1.1)
    plt.xticks(angle_list)
    plt.title(f"{net_size}by{net_size} Output Reduction")
    plt.legend()
    # plt.savefig(f"{save_path}/Nonlinearity_{net_size}by{net_size}_output_reduction_Sample{sample_freq}_Update{eval_freq}.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(7.5, 7.5))
    plt.plot(angle_list, df['omc_test'], '-o', markersize = 10, linewidth = 3, label='All Outputs')
    plt.plot(angle_list, df['omc_test_hyvx'], '-o', markersize = 10, linewidth = 2, label='Horizontal Y + Vertical X')
    plt.plot(angle_list, df['omc_test_ns'], '-o', markersize = 10, linewidth = 2, label='Near Springs')
    plt.plot(angle_list, df['omc_test_na'], '-o', markersize = 10, linewidth = 2, label='Near Actuation')
    plt.xlabel("Angle of Stimulation (deg)")
    plt.ylabel("Capacity")
    plt.ylim(-0.1, 1.1)
    plt.title(f"{net_size}by{net_size} Output Reduction")
    plt.legend()
    plt.grid(True)
    # plt.savefig(f"{save_path}/Memory_{net_size}by{net_size}_output_reduction_Sample{sample_freq}_Update{eval_freq}.pdf", dpi=300)
    plt.close()