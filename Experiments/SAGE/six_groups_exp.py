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
    stimulation_point_label = 34 #69 #
    vid_path = os.path.join(os.path.dirname(__file__), f"{net_size}by{net_size}/Videos")
    # save_path = "C:/D/ABHYAAS/OneDrive - Virginia Tech/naughtonlab - active_projects/2025-JIMSS_paper/pdf_v3/Fig 4"
    
    conns_end = net_size*net_size*2
    hors_end = conns_end + ((net_size+1)*net_size)*2

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
    alpha_mem = 1
    regressor = "Rid" #"Lin" #

    leg_x = np.linspace(1, leg_max_order, leg_max_order)
    max_timesteps_back = np.rint(max_time_back_seconds * fps/frame_rate_for_img).astype(int)
    mem_x = np.linspace(0, max_time_back_seconds, max_timesteps_back+1)

    groups_dict = {
        'All Outputs': [],
        'Conns X': [],
        'Conns Y': [],
        'Hors X': [],
        'Hors Y': [],
        'Vers X': [],
        'Vers Y': []}

    # groups_dict = {
    #     'All Outputs': [],
    #     'Concatenated': []}

    df = pd.DataFrame(columns=['Angle', 'max_onc_train_label', 'max_onc_train', 'max_onc_test_label', 'max_onc_test',
                               'max_omc_train_label', 'max_omc_train',  'max_omc_test_label', 'max_omc_test'])
    
    for i in range(len(angle_list)):
        angle = angle_list[i]

        name = f"{sample_freq}_{eval_freq}_{angle}.MP4"

        data_path = f"{vid_path}/{name}_frames"

        orig = np.load(f'{data_path}/eval_data.npz', allow_pickle=True)
        input_signal = orig['input_data']
        output_data = orig['output_data']
        vers_end = output_data.shape[1]

        # Nonlinearity Testing
        legendre_capacity_train, legendre_capacity_test, legendre_R2_train, legendre_R2_test, legendre_rmse_train, legendre_rmse_test = nonlinearity_testing(input_signal, output_data, leg_max_order, test_size, alpha_leg, regressor)
        # Memory Testing
        memory_capacity_train, memory_capacity_test, memory_R2_train, memory_R2_test, memory_rmse_train, memory_rmse_test = memory_testing(input_signal, output_data, max_timesteps_back, test_size, alpha_mem, regressor)

        onc_train = sum(legendre_capacity_train)/len(legendre_capacity_train)
        onc_test = sum(legendre_capacity_test)/len(legendre_capacity_test)
        omc_train = sum(memory_capacity_train)/len(memory_capacity_train)
        omc_test = sum(memory_capacity_test)/len(memory_capacity_test)
        print("All Outputs Done")

        output_cx = output_data[:, 0:conns_end:2]
        output_cy = output_data[:, 1:conns_end:2]
        output_hx = output_data[:, conns_end:hors_end:2]
        output_hy = output_data[:, conns_end+1:hors_end:2]
        output_vx = output_data[:, hors_end:vers_end:2]
        output_vy = output_data[:, hors_end+1:vers_end:2]
        output_concat = np.hstack([output_cx, output_cy, output_hx, output_hy, output_vx, output_vy])
        print(output_data.shape, output_concat.shape)  

        # Nonlinearity testing
        legendre_capacity_train_cx, legendre_capacity_test_cx, legendre_R2_train_cx, legendre_R2_test_cx, legendre_rmse_train_cx, legendre_rmse_test_cx = nonlinearity_testing(input_signal, output_cx, leg_max_order, test_size, alpha_leg, regressor)
        # Memory testing
        memory_capacity_train_cx, memory_capacity_test_cx, memory_R2_train_cx, memory_R2_test_cx, memory_rmse_train_cx, memory_rmse_test_cx = memory_testing(input_signal, output_cx, max_timesteps_back, test_size, alpha_mem, regressor)

        onc_train_cx = sum(legendre_capacity_train_cx)/len(legendre_capacity_train_cx)
        onc_test_cx = sum(legendre_capacity_test_cx)/len(legendre_capacity_test_cx)
        omc_train_cx = sum(memory_capacity_train_cx)/len(memory_capacity_train_cx)
        omc_test_cx = sum(memory_capacity_test_cx)/len(memory_capacity_test_cx)

        print("Conns X Done")
        
        # Nonlinearity testing
        legendre_capacity_train_cy, legendre_capacity_test_cy, legendre_R2_train_cy, legendre_R2_test_cy, legendre_rmse_train_cy, legendre_rmse_test_cy = nonlinearity_testing(input_signal, output_cy, leg_max_order, test_size, alpha_leg, regressor)
        # Memory testing
        memory_capacity_train_cy, memory_capacity_test_cy, memory_R2_train_cy, memory_R2_test_cy, memory_rmse_train_cy, memory_rmse_test_cy = memory_testing(input_signal, output_cy, max_timesteps_back, test_size, alpha_mem, regressor)
        
        onc_train_cy = sum(legendre_capacity_train_cy)/len(legendre_capacity_train_cy)
        onc_test_cy = sum(legendre_capacity_test_cy)/len(legendre_capacity_test_cy)
        omc_train_cy = sum(memory_capacity_train_cy)/len(memory_capacity_train_cy)
        omc_test_cy = sum(memory_capacity_test_cy)/len(memory_capacity_test_cy)

        print("Conns Y Done")

        # Nonlinearity testing
        legendre_capacity_train_hx, legendre_capacity_test_hx, legendre_R2_train_hx, legendre_R2_test_hx, legendre_rmse_train_hx, legendre_rmse_test_hx = nonlinearity_testing(input_signal, output_hx, leg_max_order, test_size, alpha_leg, regressor)
        # Memory testing
        memory_capacity_train_hx, memory_capacity_test_hx, memory_R2_train_hx, memory_R2_test_hx, memory_rmse_train_hx, memory_rmse_test_hx = memory_testing(input_signal, output_hx, max_timesteps_back, test_size, alpha_mem, regressor)
        
        onc_train_hx = sum(legendre_capacity_train_hx)/len(legendre_capacity_train_hx)
        onc_test_hx = sum(legendre_capacity_test_hx)/len(legendre_capacity_test_hx)
        omc_train_hx = sum(memory_capacity_train_hx)/len(memory_capacity_train_hx)
        omc_test_hx = sum(memory_capacity_test_hx)/len(memory_capacity_test_hx)

        print("Hors X Done")

        # Nonlinearity testing
        legendre_capacity_train_hy, legendre_capacity_test_hy, legendre_R2_train_hy, legendre_R2_test_hy, legendre_rmse_train_hy, legendre_rmse_test_hy = nonlinearity_testing(input_signal, output_hy, leg_max_order, test_size, alpha_leg, regressor)
        # Memory testing
        memory_capacity_train_hy, memory_capacity_test_hy, memory_R2_train_hy, memory_R2_test_hy, memory_rmse_train_hy, memory_rmse_test_hy = memory_testing(input_signal, output_hy, max_timesteps_back, test_size, alpha_mem, regressor)
        onc_train_hy = sum(legendre_capacity_train_hy)/len(legendre_capacity_train_hy)
        onc_test_hy = sum(legendre_capacity_test_hy)/len(legendre_capacity_test_hy)
        omc_train_hy = sum(memory_capacity_train_hy)/len(memory_capacity_train_hy)
        omc_test_hy = sum(memory_capacity_test_hy)/len(memory_capacity_test_hy)

        print("Hors Y Done")

        # Nonlinearity testing
        legendre_capacity_train_vx, legendre_capacity_test_vx, legendre_R2_train_vx, legendre_R2_test_vx, legendre_rmse_train_vx, legendre_rmse_test_vx = nonlinearity_testing(input_signal, output_vx, leg_max_order, test_size, alpha_leg, regressor)
        # Memory testing
        memory_capacity_train_vx, memory_capacity_test_vx, memory_R2_train_vx, memory_R2_test_vx, memory_rmse_train_vx, memory_rmse_test_vx = memory_testing(input_signal, output_vx, max_timesteps_back, test_size, alpha_mem, regressor)
        
        onc_train_vx = sum(legendre_capacity_train_vx)/len(legendre_capacity_train_vx)
        onc_test_vx = sum(legendre_capacity_test_vx)/len(legendre_capacity_test_vx)
        omc_train_vx = sum(memory_capacity_train_vx)/len(memory_capacity_train_vx)
        omc_test_vx = sum(memory_capacity_test_vx)/len(memory_capacity_test_vx)

        print("Vers X Done")
                
        # Nonlinearity testing
        legendre_capacity_train_vy, legendre_capacity_test_vy, legendre_R2_train_vy, legendre_R2_test_vy, legendre_rmse_train_vy, legendre_rmse_test_vy = nonlinearity_testing(input_signal, output_vy, leg_max_order, test_size, alpha_leg, regressor)
        # Memory testing
        memory_capacity_train_vy, memory_capacity_test_vy, memory_R2_train_vy, memory_R2_test_vy, memory_rmse_train_vy, memory_rmse_test_vy = memory_testing(input_signal, output_vy, max_timesteps_back, test_size, alpha_mem, regressor)
        
        onc_train_vy = sum(legendre_capacity_train_vy)/len(legendre_capacity_train_vy)
        onc_test_vy = sum(legendre_capacity_test_vy)/len(legendre_capacity_test_vy)
        omc_train_vy = sum(memory_capacity_train_vy)/len(memory_capacity_train_vy)
        omc_test_vy = sum(memory_capacity_test_vy)/len(memory_capacity_test_vy)

        print("Vers Y Done")

        onc_train_list = [onc_train_cx, onc_train_cy, onc_train_hx, onc_train_hy, onc_train_vx, onc_train_vy]
        onc_test_list = [onc_test_cx, onc_test_cy, onc_test_hx, onc_test_hy, onc_test_vx, onc_test_vy]
        omc_train_list = [omc_train_cx, omc_train_cy, omc_train_hx, omc_train_hy, omc_train_vx, omc_train_vy]
        omc_test_list = [omc_test_cx, omc_test_cy, omc_test_hx, omc_test_hy, omc_test_vx, omc_test_vy]

        max_onc_train = max(onc_train_list)
        max_onc_test = max(onc_test_list)
        max_omc_train = max(omc_train_list)
        max_omc_test = max(omc_test_list)
        idx_max_onc_train = onc_train_list.index(max_onc_train)
        idx_max_onc_test = onc_test_list.index(max_onc_test)
        idx_max_omc_train = omc_train_list.index(max_omc_train)
        idx_max_omc_test = omc_test_list.index(max_omc_test)

        labels_list = ['Conns X', 'Conns Y', 'Hors X', 'Hors Y', 'Vers X', 'Vers Y']

        df.at[i, 'Angle'] = angle
        df.at[i, 'max_onc_train_label'] = labels_list[idx_max_onc_train]
        df.at[i, 'max_onc_train'] = max_onc_train
        df.at[i, 'max_onc_test_label'] = labels_list[idx_max_onc_test]
        df.at[i, 'max_onc_test'] = max_onc_test
        df.at[i, 'max_omc_train_label'] = labels_list[idx_max_omc_train]
        df.at[i, 'max_omc_train'] = max_omc_train
        df.at[i, 'max_omc_test_label'] = labels_list[idx_max_omc_test]
        df.at[i, 'max_omc_test'] = max_omc_test

        onc_train_list = [onc_train] + onc_train_list
        onc_test_list = [onc_test] + onc_test_list
        omc_train_list = [omc_train] + omc_train_list
        omc_test_list = [omc_test] + omc_test_list
        labels_list = ['All Outputs'] + labels_list

        plt.figure(figsize=(7.5, 7.5))
        plt.bar(labels_list, onc_train_list, label='Training ONC')
        plt.bar(labels_list, onc_test_list, alpha=0.6, label='Testing ONC')
        plt.ylabel("Overall Nonlinear Capacity (ONC)")
        plt.title(f"Overall Nonlinear Capacity (ONC) at {angle} degrees")
        plt.ylim(0, onc_train_list[0]+0.1)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{data_path}/ONC_at_{angle}_degrees.png")
        # plt.show()
        plt.close()
        plt.figure(figsize=(7.5, 7.5))
        plt.bar(labels_list, omc_train_list, label='Training OMC')
        plt.bar(labels_list, omc_test_list, alpha=0.6, label='Testing OMC')
        plt.ylabel("Overall Memory Capacity (OMC)")
        plt.title(f"Overall Memory Capacity (OMC) at {angle} degrees")
        plt.ylim(0, omc_train_list[0]+0.1)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{data_path}/OMC_at_{angle}_degrees.png")
        # plt.show()
        plt.close()

        groups_dict['All Outputs'].append([onc_test, omc_test])
        groups_dict['Conns X'].append([onc_test_cx, omc_test_cx])
        groups_dict['Conns Y'].append([onc_test_cy, omc_test_cy])
        groups_dict['Hors X'].append([onc_test_hx, omc_test_hx])
        groups_dict['Hors Y'].append([onc_test_hy, omc_test_hy])
        groups_dict['Vers X'].append([onc_test_vx, omc_test_vx])
        groups_dict['Vers Y'].append([onc_test_vy, omc_test_vy])

    df.to_csv(f"{vid_path}/six_grps_summary_sample{sample_freq}_eval{eval_freq}.csv", index=False)
     
    plt.figure(figsize=(7.5, 7.5))
    for key in groups_dict.keys():
        onc_values = [item[0] for item in groups_dict[key]]
        plt.plot(angle_list, onc_values, '-o', label=key)
    plt.xlabel("Stimulation Angle (degrees)")
    plt.ylabel("Overall Nonlinear Capacity (ONC)")
    plt.title("Overall Nonlinear Capacity (ONC) vs Stimulation Angle")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{vid_path}/ONC_six_groups_sample{sample_freq}_eval{eval_freq}.png")
    plt.show()
    plt.close()
    plt.figure(figsize=(7.5, 7.5))
    for key in groups_dict.keys():
        omc_values = [item[1] for item in groups_dict[key]]
        plt.plot(angle_list, omc_values, '-o', label=key)
    plt.xlabel("Stimulation Angle (degrees)")
    plt.ylabel("Overall Memory Capacity (OMC)")
    plt.title("Overall Memory Capacity (OMC) vs Stimulation Angle")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{vid_path}/OMC_six_groups_sample{sample_freq}_eval{eval_freq}.png")
    plt.show()
    plt.close()
