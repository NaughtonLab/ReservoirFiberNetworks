import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn import preprocessing
from scipy.special import legendre

matplotlib.rc('pdf', fonttype=42)

def narma2_test(input, output, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    # normalizing input between 0 and 0.5
    x = output
    y = np.zeros((input.shape[0], input.shape[1]))
 
    a = 0.4
    b = 0.4
    g = 0.6
    d = 0.1
    for t in range(1, input.shape[0]-1):
        alpha_term = a * y[t]
        beta_term = b * y[t] * y[t-1]
        gamma_term = g * input[t]**3
        y[t+1] = alpha_term + beta_term + gamma_term + d

    # Ignoring first 500 samples to avoid initial transients
    x = x[200:, :]
    y = y[200:, :]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

    # Training
    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    idx = np.where(abs(y_train_pred) > 2)
    y_train_pred[idx] = np.mean(y)
    y_train[idx, 0] = np.mean(y)

    y2_train = (1/len(y_train)) * np.sum((y_train - np.mean(y_train))**2)

    MSE_train = mean_squared_error(y_train, y_train_pred)
    capacity_train = 1 - MSE_train/y2_train
    RMSE_train = root_mean_squared_error(y_true=y_train, y_pred=y_train_pred)
    R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

    if R2_train < 0:
        R2_train = 0
    if capacity_train < 0:
        capacity_train = 0
    if RMSE_train < 0:
        RMSE_train = 0

    # Testing
    y_test_pred = clf.predict(x_test)

    idx = np.where(abs(y_test_pred) > 2)
    y_test_pred[idx] = np.mean(y)
    y_test[idx, 0] = np.mean(y)

    y2_test = (1/len(y_test)) * np.sum((y_test - np.mean(y_test))**2)

    MSE_test = mean_squared_error(y_test, y_test_pred)
    capacity_test = 1 - MSE_test/y2_test
    RMSE_test = root_mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

    if R2_test < 0:
        R2_test = 0
    if capacity_test < 0:
        capacity_test = 0
    if RMSE_test < 0:
        RMSE_test = 0

    AE_list = []
    SE_list = []
    E_list = []
    NE_list = []
    for i in range(len(y_test)):
        ae = np.abs(y_test[i] - y_test_pred[i])
        se = (y_test[i] - y_test_pred[i])**2
        e = y_test[i] - y_test_pred[i]
        ne = e / np.mean(y_test)
        AE_list.append(ae)
        SE_list.append(se)
        E_list.append(e)
        NE_list.append(ne)

    np.savez("./Experiments/SAGE/6by6/Videos/2_100_50.MP4_frames/narma2_errors.npz", AE=AE_list, SE=SE_list, E=E_list, NE=NE_list)

    return capacity_train, capacity_test, R2_train, R2_test, RMSE_train, RMSE_test, y_train, y_train_pred, y_test, y_test_pred

def narma5_test(input, output, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    # normalizing input between 0 and 0.5
    x = output
    y = np.zeros((input.shape[0], input.shape[1]))

    a = 0.3
    b = 0.05
    g = 1.5
    d = 0.1
    n = 5-1
    for t in range(n, input.shape[0]-1):
        alpha_term = a * y[t]
        beta_term = b * y[t] * sum([y[t-i] for i in range(0, n+1)])
        gamma_term = g * input[t] * input[t-n]
        y[t+1] = alpha_term + beta_term + gamma_term + d

    # Ignoring first 500 samples to avoid initial transients
    x = x[200:, :]
    y = y[200:, :]

    # y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

    # Training
    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    idx = np.where(abs(y_train_pred) > 2)
    y_train_pred[idx] = np.mean(y)
    y_train[idx, 0] = np.mean(y)

    y2_train = (1/len(y_train)) * np.sum((y_train - np.mean(y_train))**2)

    MSE_train = mean_squared_error(y_train, y_train_pred)
    capacity_train = 1 - MSE_train/y2_train
    RMSE_train = root_mean_squared_error(y_true=y_train, y_pred=y_train_pred)
    R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

    if R2_train < 0:
        R2_train = 0
    if capacity_train < 0:
        capacity_train = 0
    if RMSE_train < 0:
        RMSE_train = 0

    # Testing
    y_test_pred = clf.predict(x_test)

    idx = np.where(abs(y_test_pred) > 2)
    y_test_pred[idx] = np.mean(y)
    y_test[idx, 0] = np.mean(y)

    y2_test = (1/len(y_test)) * np.sum((y_test - np.mean(y_test))**2)

    MSE_test = mean_squared_error(y_test, y_test_pred)
    capacity_test = 1 - MSE_test/y2_test
    RMSE_test = root_mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

    if R2_test < 0:
        R2_test = 0
    if capacity_test < 0:
        capacity_test = 0
    if RMSE_test < 0:
        RMSE_test = 0

    AE_list = []
    SE_list = []
    E_list = []
    NE_list = []
    for i in range(len(y_test)):
        ae = np.abs(y_test[i] - y_test_pred[i])
        se = (y_test[i] - y_test_pred[i])**2
        e = y_test[i] - y_test_pred[i]
        ne = e / np.mean(y_test)
        AE_list.append(ae)
        SE_list.append(se)
        E_list.append(e)
        NE_list.append(ne)
    
    np.savez("./Experiments/SAGE/6by6/Videos/2_100_50.MP4_frames/narma5_errors.npz", AE=AE_list, SE=SE_list, E=E_list, NE=NE_list)

    return capacity_train, capacity_test, R2_train, R2_test, RMSE_train, RMSE_test, y_train, y_train_pred, y_test, y_test_pred

def narma10_test(input, output, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    # normalizing input between 0 and 0.5
    x = output
    y = np.zeros((input.shape[0], input.shape[1]))

    a = 0.3
    b = 0.05
    g = 1.5
    d = 0.1
    n = 10-1
    for t in range(n, input.shape[0]-1):
        alpha_term = a * y[t]
        beta_term = b * y[t] * sum([y[t-i] for i in range(0, n+1)])
        gamma_term = g * input[t] * input[t-n]
        y[t+1] = alpha_term + beta_term + gamma_term + d

    # Ignoring first 200 samples to avoid initial transients
    x = x[200:, :]
    y = y[200:, :]

    # y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)

    # Training
    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    idx = np.where(abs(y_train_pred) > 2)
    y_train_pred[idx] = np.mean(y)
    y_train[idx, 0] = np.mean(y)

    y2_train = (1/len(y_train)) * np.sum((y_train - np.mean(y_train))**2)

    MSE_train = mean_squared_error(y_train, y_train_pred)
    capacity_train = 1 - MSE_train/y2_train
    RMSE_train = root_mean_squared_error(y_true=y_train, y_pred=y_train_pred)
    R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

    if R2_train < 0:
        R2_train = 0
    if capacity_train < 0:
        capacity_train = 0
    if RMSE_train < 0:
        RMSE_train = 0

    # Testing
    y_test_pred = clf.predict(x_test)

    idx = np.where(abs(y_test_pred) > 2)
    y_test_pred[idx] = np.mean(y)
    y_test[idx, 0] = np.mean(y)

    y2_test = (1/len(y_test)) * np.sum((y_test - np.mean(y_test))**2)

    MSE_test = mean_squared_error(y_test, y_test_pred)
    capacity_test = 1 - MSE_test/y2_test
    RMSE_test = root_mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

    if R2_test < 0:
        R2_test = 0
    if capacity_test < 0:
        capacity_test = 0
    if RMSE_test < 0:
        RMSE_test = 0

    AE_list = []
    SE_list = []
    E_list = []
    NE_list = []
    for i in range(len(y_test)):
        ae = np.abs(y_test[i] - y_test_pred[i])
        se = (y_test[i] - y_test_pred[i])**2
        e = y_test[i] - y_test_pred[i]
        ne = e / np.mean(y_test)
        AE_list.append(ae)
        SE_list.append(se)
        E_list.append(e)
        NE_list.append(ne)

    np.savez("./Experiments/SAGE/6by6/Videos/2_100_50.MP4_frames/narma10_errors.npz", AE=AE_list, SE=SE_list, E=E_list, NE=NE_list)

    return capacity_train, capacity_test, R2_train, R2_test, RMSE_train, RMSE_test, y_train, y_train_pred, y_test, y_test_pred

if __name__ == "__main__":
    save_path = "C:/D/ABHYAAS/OneDrive - Virginia Tech/naughtonlab - active_projects/2025-JIMSS_paper/pdf_v4/Fig 4"
    net_size = 6 #4 #    
    stimulation_point_label = 69 #34 #
    vid_path = os.path.join(os.path.dirname(__file__), f"{net_size}by{net_size}/Videos")
    # save_path = vid_path

    sample_freq = 2
    eval_freq = 100
    angle_list = [50] #[25, 30, 35, 40, 45, 50] #

    test_size = 0.25
    
    leg_max_order = 10
    # max_timesteps_back = 10
    rotate = False
    max_time_back_seconds = 1
    fps = 120
    fps_desired = 10
    frame_rate_for_img = 1

    original_dt = 1 / fps
    desired_dt = 1 / fps_desired
    downsample_factor = int(desired_dt / original_dt)
    print(f"Original dt: {original_dt}, Desired dt: {desired_dt}, Downsample factor: {downsample_factor}")

    alpha = 1
    regressor = "Rid"

    df = pd.DataFrame(columns=['sample_freq', 'eval_freq', 'angle', 'n2_cap_train', 'n2_cap_test', 'n5_cap_train', 'n5_cap_test', 'n10_cap_train', 'n10_cap_test'])

    n2_R2_train_scores_list = []
    n2_R2_test_scores_list = []
    n2_capacity_train_scores_list = []
    n2_capacity_test_scores_list = []
    n2_RMSE_train_scores_list = []
    n2_RMSE_test_scores_list = []

    n5_R2_train_scores_list = []
    n5_R2_test_scores_list = []
    n5_capacity_train_scores_list = []
    n5_capacity_test_scores_list = []
    n5_RMSE_train_scores_list = []
    n5_RMSE_test_scores_list = []

    n10_R2_train_scores_list = []
    n10_R2_test_scores_list = []
    n10_capacity_train_scores_list = []
    n10_capacity_test_scores_list = []
    n10_RMSE_train_scores_list = []
    n10_RMSE_test_scores_list = []
    
    for i in range(len(angle_list)):
        angle = angle_list[i]

        name = f"{sample_freq}_{eval_freq}_{angle}.MP4"

        data_path = f"{vid_path}/{name}_frames"

        data = np.load(f"{data_path}/eval_data.npz", allow_pickle=True)
        input_data = data['input_data']
        output_data = data['output_data']

        input_data_ds = input_data[::downsample_factor]
        output_data_ds = output_data[::downsample_factor]

        time_min = 0
        time_max = input_data.shape[0] * original_dt
        # time_axis = np.arange(time_min, time_max, original_dt)
        # time_axis_test_set = time_axis[int(input_data.shape[0]*(1 - test_size)):]
        time_axis_ds = np.arange(time_min, time_max, desired_dt)
        time_axis_ds_test_set = time_axis_ds[int(input_data_ds.shape[0]*(1 - test_size)):]

        # Normalize input between 0 and 0.2
        input_data_narma2_5 = 0 + (input_data_ds - np.min(input_data_ds)) / (np.max(input_data_ds) - np.min(input_data_ds)) * (0.2 - (0))
        input_data_narma10 = 0 + (input_data_ds - np.min(input_data_ds)) / (np.max(input_data_ds) - np.min(input_data_ds)) * (0.2 - (0))
        # Normalize output
        output_data_ds /= np.mean(output_data_ds, axis=0)
        output_data_ds /= np.std(output_data_ds, axis=0)
        
        n2_cap_train, n2_cap_test, n2_R2_train, n2_R2_test, n2_RMSE_train, n2_RMSE_test, n2_y_train, n2_y_train_pred, n2_y_test, n2_y_test_pred = narma2_test(input_data_narma2_5, output_data_ds, regressor, test_size, alpha)
        n5_cap_train, n5_cap_test, n5_R2_train, n5_R2_test, n5_RMSE_train, n5_RMSE_test, n5_y_train, n5_y_train_pred, n5_y_test, n5_y_test_pred = narma5_test(input_data_narma2_5, output_data_ds, regressor, test_size, alpha)
        n10_cap_train, n10_cap_test, n10_R2_train, n10_R2_test, n10_RMSE_train, n10_RMSE_test, n10_y_train, n10_y_train_pred, n10_y_test, n10_y_test_pred = narma10_test(input_data_narma10, output_data_ds, regressor, test_size, alpha)

        print(f"Angle: {angle} degrees - NARMA-2 Capacity Train: {n2_cap_train}, Test: {n2_cap_test}, NARMA-5 Capacity Train: {n5_cap_train}, Test: {n5_cap_test}, NARMA-10 Capacity Train: {n10_cap_train}, Test: {n10_cap_test}")

        if net_size == 6 and angle == 50:
            plt.figure(figsize=(10, 5))
            plt.plot(time_axis_ds_test_set[:len(n2_y_test)], n2_y_test, label='True')
            plt.plot(time_axis_ds_test_set[:len(n2_y_test_pred)], n2_y_test_pred, '--', label='Predicted')
            plt.title(f'NARMA-2 Test Output')
            plt.xlabel('Time (s)')
            plt.ylabel('Output')
            plt.legend()
            # plt.savefig(f"{save_path}/TestSet_NARMA2_{fps_desired}Hz_{net_size}by{net_size}_{angle}deg.pdf", dpi=300)
            plt.close()
            plt.figure(figsize=(10, 5))
            plt.plot(time_axis_ds_test_set[:len(n5_y_test)], n5_y_test, label='True')
            plt.plot(time_axis_ds_test_set[:len(n5_y_test_pred)], n5_y_test_pred, '--', label='Predicted')
            plt.title(f'NARMA-5 Test Output')
            plt.xlabel('Time (s)')
            plt.ylabel('Output')
            plt.legend()
            # plt.savefig(f"{save_path}/TestSet_NARMA5_{fps_desired}Hz_{net_size}by{net_size}_{angle}deg.pdf", dpi=300)
            plt.close()
            plt.figure(figsize=(10, 5))
            plt.plot(time_axis_ds_test_set[:len(n10_y_test)], n10_y_test, label='True')
            plt.plot(time_axis_ds_test_set[:len(n10_y_test_pred)], n10_y_test_pred, '--', label='Predicted')
            plt.title(f'NARMA-10 Test Output')
            plt.xlabel('Time (s)')
            plt.ylabel('Output')
            plt.legend()
            # plt.savefig(f"{save_path}/TestSet_NARMA10_{fps_desired}Hz_{net_size}by{net_size}_{angle}deg.pdf", dpi=300)
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(time_axis_ds[200:], np.concatenate([n2_y_train, n2_y_test]), label='True')
            plt.plot(time_axis_ds[200:], np.concatenate([n2_y_train_pred, n2_y_test_pred]), '--', label='Predicted')
            plt.title(f'NARMA-2 Test Output')
            plt.xlabel('Time (s)')
            plt.ylabel('Output')
            plt.legend()
            # plt.savefig(f"{save_path}/NARMA2_{fps_desired}Hz_{net_size}by{net_size}_{angle}deg.pdf", dpi=300)
            plt.close()
            plt.figure(figsize=(10, 5))
            plt.plot(time_axis_ds[200:], np.concatenate([n5_y_train, n5_y_test]), label='True')
            plt.plot(time_axis_ds[200:], np.concatenate([n5_y_train_pred, n5_y_test_pred]), '--', label='Predicted')
            plt.title(f'NARMA-5 Test Output')
            plt.xlabel('Time (s)')
            plt.ylabel('Output')
            plt.legend()
            # plt.savefig(f"{save_path}/NARMA5_{fps_desired}Hz_{net_size}by{net_size}_{angle}deg.pdf", dpi=300)
            plt.close()
            plt.figure(figsize=(10, 5))
            plt.plot(time_axis_ds[200:], np.concatenate([n10_y_train, n10_y_test]), label='True')
            plt.plot(time_axis_ds[200:], np.concatenate([n10_y_train_pred, n10_y_test_pred]), '--', label='Predicted')
            plt.title(f'NARMA-10 Test Output')
            plt.xlabel('Time (s)')
            plt.ylabel('Output')
            plt.legend()
            # plt.savefig(f"{save_path}/NARMA10_{fps_desired}Hz_{net_size}by{net_size}_{angle}deg.pdf", dpi=300)
            plt.close()

        df.at[i, 'sample_freq'] = sample_freq
        df.at[i, 'eval_freq'] = eval_freq
        df.at[i, 'angle'] = angle
        df.at[i, 'n2_cap_train'] = n2_cap_train
        df.at[i, 'n2_cap_test'] = n2_cap_test
        df.at[i, 'n5_cap_train'] = n5_cap_train
        df.at[i, 'n5_cap_test'] = n5_cap_test
        df.at[i, 'n10_cap_train'] = n10_cap_train
        df.at[i, 'n10_cap_test'] = n10_cap_test

        n2_R2_train_scores_list.append(n2_R2_train)
        n2_R2_test_scores_list.append(n2_R2_test)
        n2_capacity_train_scores_list.append(n2_cap_train)
        n2_capacity_test_scores_list.append(n2_cap_test)
        n2_RMSE_train_scores_list.append(n2_RMSE_train)
        n2_RMSE_test_scores_list.append(n2_RMSE_test)
        n5_R2_train_scores_list.append(n5_R2_train)
        n5_R2_test_scores_list.append(n5_R2_test)
        n5_capacity_train_scores_list.append(n5_cap_train)
        n5_capacity_test_scores_list.append(n5_cap_test)
        n5_RMSE_train_scores_list.append(n5_RMSE_train)
        n5_RMSE_test_scores_list.append(n5_RMSE_test)
        n10_R2_train_scores_list.append(n10_R2_train)
        n10_R2_test_scores_list.append(n10_R2_test)
        n10_capacity_train_scores_list.append(n10_cap_train)
        n10_capacity_test_scores_list.append(n10_cap_test)
        n10_RMSE_train_scores_list.append(n10_RMSE_train)
        n10_RMSE_test_scores_list.append(n10_RMSE_test)

    # print(f"Capacity NARMA-2 at 50 degrees {n2_capacity_test_scores_list[-1]}")
    # print(f"R2 NARMA-2 at 50 degrees {n2_R2_test_scores_list[-1]}")
    print(f"RMSE NARMA-2 at 50 degrees {n2_RMSE_test_scores_list[-1]}")
    # print(f"Capacity NARMA-5 at 50 degrees {n5_capacity_test_scores_list[-1]}")
    # print(f"R2 NARMA-5 at 50 degrees {n5_R2_test_scores_list[-1]}")
    print(f"RMSE NARMA-5 at 50 degrees {n5_RMSE_test_scores_list[-1]}")
    # print(f"Capacity NARMA-10 at 50 degrees {n10_capacity_test_scores_list[-1]}")
    # print(f"R2 NARMA-10 at 50 degrees {n10_R2_test_scores_list[-1]}")
    print(f"RMSE NARMA-10 at 50 degrees {n10_RMSE_test_scores_list[-1]}")

    violin_data = {'n2_cap': n2_capacity_test_scores_list,
                   'n5_cap': n5_capacity_test_scores_list,
                   'n10_cap': n10_capacity_test_scores_list,
                   'n2_R2': n2_R2_test_scores_list,
                   'n5_R2': n5_R2_test_scores_list,
                   'n10_R2': n10_R2_test_scores_list,
                   'n2_RMSE': n2_RMSE_test_scores_list,
                   'n5_RMSE': n5_RMSE_test_scores_list,
                   'n10_RMSE': n10_RMSE_test_scores_list}
    
    df_violin = pd.DataFrame(violin_data)
    # df_violin.to_csv(f"{vid_path}/narma_violin_data_{sample_freq}_{eval_freq}_{fps_desired}.csv", index=False)
        
    # fignarma, axnarma= plt.subplots(1, 3, figsize=(17.5, 7.5))
    # axnarma[0].plot(angle_list, n2_capacity_train_scores_list, '-o', markersize = 10, linewidth = 3)
    # axnarma[0].plot(angle_list, n2_capacity_test_scores_list, '--o', markersize = 10, linewidth = 3)
    # axnarma[0].plot(angle_list, n2_R2_train_scores_list, '-o', markersize = 10, linewidth = 2)
    # axnarma[0].plot(angle_list, n2_R2_test_scores_list, '--o', markersize = 10, linewidth = 2)
    # axnarma[0].plot(angle_list, n2_RMSE_train_scores_list, '-o', markersize = 10, linewidth = 2)
    # axnarma[0].plot(angle_list, n2_RMSE_test_scores_list, '--o', markersize = 10, linewidth = 2)
    # axnarma[0].set_title(f'NARMA-2')
    # axnarma[0].set_xlabel('Angle (degrees)')
    # axnarma[0].set_ylabel('Score')
    # axnarma[0].legend(['Train cap', 'Test cap', 'Train R2', 'Test R2', 'Train RMSE', 'Test RMSE'])
    # axnarma[1].plot(angle_list, n5_capacity_train_scores_list, '-o', markersize = 10, linewidth = 3)
    # axnarma[1].plot(angle_list, n5_capacity_test_scores_list, '--o', markersize = 10, linewidth = 3)
    # axnarma[1].plot(angle_list, n5_R2_train_scores_list, '-o', markersize = 10, linewidth = 2)
    # axnarma[1].plot(angle_list, n5_R2_test_scores_list, '--o', markersize = 10, linewidth = 2)
    # axnarma[1].plot(angle_list, n10_RMSE_train_scores_list, '-o', markersize = 10, linewidth = 2)
    # axnarma[1].plot(angle_list, n10_RMSE_test_scores_list, '--o', markersize = 10, linewidth = 2)
    # axnarma[1].set_title(f'NARMA-5')
    # axnarma[1].set_xlabel('Angle (degrees)')
    # axnarma[1].set_ylabel('Score')
    # axnarma[1].legend(['Train cap', 'Test cap', 'Train R2', 'Test R2', 'Train RMSE', 'Test RMSE'])
    # axnarma[2].plot(angle_list, n10_capacity_train_scores_list, '-o', markersize = 10, linewidth = 3)
    # axnarma[2].plot(angle_list, n10_capacity_test_scores_list, '--o', markersize = 10, linewidth = 3)
    # axnarma[2].plot(angle_list, n5_RMSE_train_scores_list, '-o', markersize = 10, linewidth = 2)
    # axnarma[2].plot(angle_list, n5_RMSE_test_scores_list, '--o', markersize = 10, linewidth = 2)
    # axnarma[2].plot(angle_list, n10_R2_train_scores_list, '-o', markersize = 10, linewidth = 2)
    # axnarma[2].plot(angle_list, n10_R2_test_scores_list, '--o', markersize = 10, linewidth = 2)
    # axnarma[2].set_title(f'NARMA-10')
    # axnarma[2].set_xlabel('Angle (degrees)')
    # axnarma[2].set_ylabel('Score')
    # axnarma[2].legend(['Train cap', 'Test cap', 'Train R2', 'Test R2', 'Train RMSE', 'Test RMSE'])
    # for ax in axnarma:
    #     ax.set_ylim(-0.1, 1.1)
    # fignarma.suptitle(f'NARMA Performance for {net_size}by{net_size} Network Downsampled at {fps_desired} Hz')
    # fignarma.savefig(f"{save_path}/narma_performance_{sample_freq}_{eval_freq}_{fps_desired}.png", dpi=300)

    # df.to_csv(f"{save_path}/narma_performance_{sample_freq}_{eval_freq}_{fps_desired}.csv", index=False)







        