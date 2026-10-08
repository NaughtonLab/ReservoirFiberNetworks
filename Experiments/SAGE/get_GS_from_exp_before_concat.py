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

    # save_path = "C:/D/ABHYAAS/OneDrive - Virginia Tech/naughtonlab - active_projects/2025-JIMSS_paper/pdf_v3/Fig 4"
    
    net_size = 6 #4 #    
    stimulation_point_label = 69 #34 #
    vid_path = os.path.join(os.path.dirname(__file__), f"{net_size}by{net_size}/Videos")
    save_path = f"{vid_path}/corrected"

    if net_size == 4:
        c_labels = [i for i in range(5, 11+2, 2)] + [i for i in range(18, 24+2, 2)] + [i for i in range(31, 37+2, 2)] + [i for i in range(44, 50+2, 2)]
        h_labels = [i for i in range(4, 12+2, 2)] + [i for i in range(17, 25+2, 2)] + [i for i in range(30, 38+2, 2) if i != 34] + [i for i in range(43, 51+2, 2)]
        v_labels = [i for i in range(0, 3+1)] + [i for i in range(13, 16+1)] + [i for i in range(26, 29+1)] + [i for i in range(39, 42+1)] + [i for i in range(52, 55+1)]
    elif net_size == 6:
        c_labels = [i for i in range(7, 17+2, 2)] + [i for i in range(26, 36+2, 2)] + [i for i in range(45, 55+2, 2)] + [i for i in range(64, 74+2, 2)] + [i for i in range(83, 93+2, 2)] + [i for i in range(102, 112+2, 2)]
        h_labels = [i for i in range(6, 18+2, 2)] + [i for i in range(25, 37+2, 2)] + [i for i in range(44, 56+2, 2)] + [i for i in range(63, 75+2, 2) if i != 69] + [i for i in range(82, 94+2, 2)] + [i for i in range(101, 113+2, 2)]
        v_labels = [i for i in range(0, 5+1)] + [i for i in range(19, 24+1)] + [i for i in range(38, 43+1)] + [i for i in range(57, 62+1)] + [i for i in range(76, 81+1)] + [i for i in range(95, 100+1)] + [i for i in range(114, 119+1)]

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
    max_timesteps_back = np.rint(max_time_back_seconds * fps/frame_rate_for_img).astype(int)

    alpha_leg = 1
    alpha_mem = 1
    regressor = "Rid" #"Lin" #

    df = pd.DataFrame(columns=['sample_freq', 'eval_freq', 'angle', 'onc_train', 'omc_train', 'onc_test', 'omc_test'])

    for i in range(len(angle_list)):
        angle = angle_list[i]

        name = f"{sample_freq}_{eval_freq}_{angle}.MP4"

        data_path = f"{vid_path}/{name}_frames"

        clean_data = pd.read_csv(f'{data_path}/clean_data_final.csv')
        clean_data['Time_s'] = clean_data['Time']/120
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
        clean_data.to_csv(f'{data_path}/clean_data_final.csv', index=False)

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        # for j in range(min_label, max_label+1):
        #     label = j
        #     specific_label = clean_data[clean_data['Label'] == label]

        #     ax1.plot(specific_label['Relative X'], specific_label['Time_s'], label=label)
        #     ax2.plot(specific_label['Time_s'], specific_label['Relative Y'], label=label)
        #     ax3.plot(specific_label['Relative X'], specific_label['Relative Y'], label=label)

        # ax1.set_xlabel("Relative X position ($px$)")
        # ax1.set_ylabel("Time ($s$)")
        # ax1.grid(True)
        # # ax1.legend()
        # ax1.set_title("Relative X positions over time")

        # ax2.set_xlabel("Time ($s$)")
        # ax2.set_ylabel("Y position ($px$)")
        # ax2.grid(True)
        # # ax2.invert_yaxis()
        # # ax2.legend()
        # ax2.set_title("Relative Y positions over time")

        # ax3.set_xlabel("Relative X position ($mm$)")
        # ax3.set_ylabel("Y position ($mm$)")
        # ax3.grid(True)
        # # ax3.invert_yaxis()
        # # ax3.legend()
        # ax3.set_title("Relative Trajectories of all points")

        # plt.savefig(f"{data_path}/relative_trajectories.png")
        # plt.close()

        stimulation_point = clean_data[clean_data['Label'] == stimulation_point_label]
        min_shape = np.min([len(clean_data[clean_data['Label'] == i]) for i in range(min_label, max_label+1)])
        # print(min_shape)

        output_conns = None
        output_hors = None
        output_vers = None
        for j in range(min_label, max_label+1):
            if j != stimulation_point_label:
                one_over_time = clean_data[clean_data['Label']==j]
                one_over_time = one_over_time.iloc[:, 5:] - one_over_time.iloc[0, 5:]
                array = one_over_time.to_numpy()
                array = array[:min_shape, :]
                if j in c_labels:
                    if output_conns is None:
                        output_conns = array
                    else:
                        output_conns = np.hstack([output_conns, array])
                elif j in h_labels:
                    if output_hors is None:
                        output_hors = array
                    else:
                        output_hors = np.hstack([output_hors, array])
                elif j in v_labels:
                    if output_vers is None:
                        output_vers = array
                    else:
                        output_vers = np.hstack([output_vers, array])
                # if output is None:
                #     output = array
                # else:
                #     output = np.hstack([output, array])

        output = np.hstack([output_conns, output_hors, output_vers])
        output = np.nan_to_num(output, nan=0)
        output /= np.mean(output, axis=0)
        output /= np.std(output, axis=0)
        scaler = preprocessing.StandardScaler()
        output = scaler.fit_transform(output)

        input_signal = stimulation_point['Relative Y'].to_numpy()
        input_signal = input_signal[:min_shape]
        input_signal = input_signal.reshape(-1, 1)
        input_signal = np.nan_to_num(input_signal, nan=0)
        # plt.figure(figsize=(10, 5))
        # plt.plot(input_signal, '-o', label='Input signal')
        # plt.grid(True)
        # plt.xlabel("Time ($s$)")
        # plt.ylabel("Displacement of stimulation point (mm)")
        # plt.title("Input signal over time")
        # plt.savefig(f"{data_path}/input_signal.png")
        # plt.close()

        # Normalize input signal between -1 and 1
        input_signal = -1 + (input_signal - np.min(input_signal)) * (1 - (-1)) / (np.max(input_signal) - np.min(input_signal))

        if i == 1 and eval_freq == 2 and net_size == 6:
            transience = 0
        else:
            transience = 500
        
        input_signal = input_signal[transience:, :]
        output = output[transience:, :]
        print(output.shape, input_signal.shape)
        
        # plt.figure(figsize=(10, 5))
        # plt.plot(input_signal, '-o', label='Input signal')
        # plt.grid(True)
        # plt.xlabel("Time ($s$)")
        # plt.ylabel("Displacement of stimulation point (mm)")
        # plt.title("Input signal over time")
        # plt.savefig(f"{data_path}/input_signal_normalized.png")
        # plt.close()        

        matplotlib.rc('pdf', fonttype=42)

        # Nonlinearity testing
        legendre_capacity_train, legendre_capacity_test, legendre_R2_train, legendre_R2_test, legendre_rmse_train, legendre_rmse_test = nonlinearity_testing(input_signal, output, leg_max_order, test_size, alpha_leg, regressor)
        
        # Plotting the results
        # plt.figure(figsize=(7.5, 7.5))
        # plt.plot(np.linspace(1, leg_max_order, leg_max_order), legendre_capacity_train, '-o', markersize = 10, linewidth = 2, label='Training cap')
        # plt.plot(np.linspace(1, leg_max_order, leg_max_order), legendre_capacity_test, '-o', markersize = 10, linewidth = 2, label='Testing cap')
        # plt.plot(np.linspace(1, leg_max_order, leg_max_order), legendre_R2_train, '-s', markersize = 10, linewidth = 2, label='Training R2')
        # plt.plot(np.linspace(1, leg_max_order, leg_max_order), legendre_R2_test, '-s', markersize = 10, linewidth = 2, label='Testing R2')
        # plt.xlabel("Legendre Polynomial Order")
        # plt.ylabel("Capacity")
        # plt.ylim(-0.1, 1.1)
        # plt.title(f"Nonlinear Capacity {sum(legendre_capacity_train)/len(legendre_capacity_train):.4f}, {sum(legendre_capacity_test)/len(legendre_capacity_test):.4f}")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f"{data_path}/nonlinear_capacity.png")
        # plt.show()

        # Memory testing       
        memory_capacity_train, memory_capacity_test, memory_R2_train, memory_R2_test, memory_rmse_train, memory_rmse_test = memory_testing(input_signal, output, max_timesteps_back, test_size, alpha_mem, regressor)
                
        # Plotting the results
        # plt.figure(figsize=(7.5, 7.5))
        # plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), memory_capacity_train, '-o', markersize = 10, linewidth = 2, label='Training capacity')
        # plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), memory_capacity_test, '-o', markersize = 10, linewidth = 2, label='Testing capacity')
        # plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), memory_R2_train, '-s', markersize = 10, linewidth = 2, label='Training R2')
        # plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), memory_R2_test, '-s', markersize = 10, linewidth = 2, label='Testing R2')
        # plt.xlabel("Seconds in the past")
        # plt.ylabel("Capacity")
        # plt.ylim(-0.1, 1.1)
        # plt.title(f"Memory Capacity {sum(memory_capacity_train)/len(memory_capacity_train):.4f}, {sum(memory_capacity_test)/len(memory_capacity_test):.4f}")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f"{data_path}/memory_capacity.png")
        # plt.show()

        print(f"Overall Nonlinearity Capacity: {sum(legendre_capacity_test)/len(legendre_capacity_test)}, Overall Memory Capacity: {sum(memory_capacity_test)/len(memory_capacity_test)}")

        np.savez(f"{data_path}/eval_data.npz",
                 input_data=input_signal,
                 output_data=output,
                 legendre_capacity_test=legendre_capacity_test,
                 memory_capacity_test=memory_capacity_test,
                 legendre_R2_test=legendre_R2_test,
                 memory_R2_test=memory_R2_test,
                 legendre_rmse_test=legendre_rmse_test,
                 memory_rmse_test=memory_rmse_test)
        
        df.at[i, 'sample_freq'] = sample_freq
        df.at[i, 'eval_freq'] = eval_freq
        df.at[i, 'angle'] = angle
        df.at[i, 'onc_train'] = sum(legendre_capacity_train)/len(legendre_capacity_train)
        df.at[i, 'omc_train'] = sum(memory_capacity_train)/len(memory_capacity_train)
        df.at[i, 'onc_test'] = sum(legendre_capacity_test)/len(legendre_capacity_test)
        df.at[i, 'omc_test'] = sum(memory_capacity_test)/len(memory_capacity_test)

    df.to_csv(f"{vid_path}/grid_search_results_{sample_freq}_{eval_freq}_alpha_leg{alpha_leg}_alpha_mem{alpha_mem}.csv", index=False)

    plt.figure(figsize=(7.5, 7.5))
    plt.plot(angle_list, df['onc_test'], '-o', markersize = 10, linewidth = 2)
    plt.xlabel("Angle of Stimulation (deg)")
    plt.ylabel("Capacity")
    plt.ylim(-0.1, 1.1)
    plt.xticks(angle_list)
    plt.title(f"Nonlinear Capacity: Sample Freq: {sample_freq} Hz, Update Freq: {eval_freq} Hz")
    plt.savefig(f"{save_path}/Nonlinearity_{net_size}by{net_size}_grid_search_Sample{sample_freq}_Update{eval_freq}.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(7.5, 7.5))
    plt.plot(angle_list, df['omc_test'], '-o', markersize = 10, linewidth = 2)
    plt.xlabel("Angle of Stimulation (deg)")
    plt.ylabel("Capacity")
    plt.ylim(-0.1, 1.1)
    plt.xticks(angle_list)
    plt.title(f"Memory Capacity: Sample Freq: {sample_freq} Hz, Update Freq: {eval_freq} Hz")
    plt.savefig(f"{save_path}/Memory_{net_size}by{net_size}_grid_search_Sample{sample_freq}_Update{eval_freq}.pdf", dpi=300)
    plt.close()    
    plt.show()