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

matplotlib.rc('pdf', fonttype=42)

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

def near_springs_selection_6(conns, hors, vers, net_size):
    near_springs_outputs = []

    hor_idx = 0
    ver_idx = 0

    for i in range(0, conns.shape[1], 2):
        if hor_idx == 0 and ver_idx == 5:
            near_springs_outputs.append(conns[:, i])
            near_springs_outputs.append(conns[:, i+1])
        elif hor_idx == 1 and ver_idx >= 4:
            near_springs_outputs.append(conns[:, i])
            near_springs_outputs.append(conns[:, i+1])
        elif hor_idx == 2 and ver_idx >= 3:
            near_springs_outputs.append(conns[:, i])
            near_springs_outputs.append(conns[:, i+1])
        elif hor_idx == 3 and ver_idx >= 2:
            near_springs_outputs.append(conns[:, i])
            near_springs_outputs.append(conns[:, i+1])
        elif hor_idx == 4 and ver_idx >= 1:
            near_springs_outputs.append(conns[:, i])
            near_springs_outputs.append(conns[:, i+1])
        elif hor_idx == 5:
            near_springs_outputs.append(conns[:, i])
            near_springs_outputs.append(conns[:, i+1])

        ver_idx += 1
        if ver_idx == net_size:
            ver_idx = 0
            hor_idx += 1

    hor_idx = 0
    ver_idx = 0

    for i in range(0, hors.shape[1], 2):
        if hor_idx == 0 and ver_idx >= 5:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])
        elif hor_idx == 1 and ver_idx >= 5:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])
        elif hor_idx == 2 and ver_idx >= 3:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])
        elif hor_idx == 3 and ver_idx >= 3:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])
            if ver_idx == 5:
                ver_idx += 1
        elif hor_idx == 4 and ver_idx >= 2:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])
        elif hor_idx == 5 and ver_idx >= 1:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])

        ver_idx += 1
        if ver_idx == net_size + 1:
            ver_idx = 0
            hor_idx += 1

    hor_idx = 0
    ver_idx = 0

    for i in range(vers.shape[1]):
        if ver_idx == 0 and hor_idx >= 5:
            near_springs_outputs.append(vers[:, i])
            near_springs_outputs.append(vers[:, i+1])
        elif ver_idx == 1 and hor_idx >= 5:
            near_springs_outputs.append(vers[:, i])
            near_springs_outputs.append(vers[:, i+1])
        elif ver_idx == 2 and hor_idx >= 3:
            near_springs_outputs.append(vers[:, i])
            near_springs_outputs.append(vers[:, i+1])
        elif ver_idx == 3 and hor_idx >= 3:
            near_springs_outputs.append(vers[:, i])
            near_springs_outputs.append(vers[:, i+1])
        elif ver_idx == 4 and hor_idx >= 2:
            near_springs_outputs.append(vers[:, i])
            near_springs_outputs.append(vers[:, i+1])
        elif ver_idx == 5 and hor_idx >= 1:
            near_springs_outputs.append(vers[:, i])
            near_springs_outputs.append(vers[:, i+1])
        
        hor_idx += 1
        if hor_idx == net_size + 1:
            hor_idx = 0
            ver_idx += 1

    near_springs_outputs_arr = np.hstack([arr[:, np.newaxis] for arr in near_springs_outputs])

    return near_springs_outputs_arr

def near_actuation_selection_6(conns, hors, vers, net_size):
    near_actuation_outputs = []

    hor_idx = 0
    ver_idx = 0

    near_actuation_outputs = []

    hor_idx = 0
    ver_idx = 0

    for i in range(0, conns.shape[1], 2):
        if (hor_idx >= 1 and hor_idx <= 4) and (ver_idx >= 1 and ver_idx <= 4):
            near_actuation_outputs.append(conns[:, i])
            near_actuation_outputs.append(conns[:, i+1])

        ver_idx += 1
        if ver_idx == net_size:
            ver_idx = 0
            hor_idx += 1

    hor_idx = 0
    ver_idx = 0

    for i in range(0, hors.shape[1], 2):
        if (hor_idx >= 1 and hor_idx <= 4) and (ver_idx >= 2 and ver_idx <= 4):
            near_actuation_outputs.append(hors[:, i])
            near_actuation_outputs.append(hors[:, i+1])
            if hor_idx == 3 and ver_idx == 5:
                ver_idx += 1

        ver_idx += 1
        if ver_idx == net_size + 1:
            ver_idx = 0
            hor_idx += 1

    hor_idx = 0
    ver_idx = 0

    for i in range(0, vers.shape[1], 2):
        if (ver_idx >= 1 and ver_idx <= 4) and (hor_idx >= 2 and hor_idx <= 4):
            near_actuation_outputs.append(vers[:, i])
            near_actuation_outputs.append(vers[:, i+1])
        
        hor_idx += 1
        if hor_idx == net_size + 1:
            hor_idx = 0
            ver_idx += 1

    near_actuation_outputs_arr = np.hstack([arr[:, np.newaxis] for arr in near_actuation_outputs])

    return near_actuation_outputs_arr

def near_springs_selection_4(conns, hors, vers, net_size):
    near_springs_outputs = []

    hor_idx = 0
    ver_idx = 0

    for i in range(0, conns.shape[1], 2):
        if hor_idx == 0 and ver_idx == 3:
            near_springs_outputs.append(conns[:, i])
            near_springs_outputs.append(conns[:, i+1])
        elif hor_idx == 1 and ver_idx >= 1:
            near_springs_outputs.append(conns[:, i])
            near_springs_outputs.append(conns[:, i+1])
        elif hor_idx == 2 and ver_idx >= 1:
            near_springs_outputs.append(conns[:, i])
            near_springs_outputs.append(conns[:, i+1])
        elif hor_idx == 3:
            near_springs_outputs.append(conns[:, i])
            near_springs_outputs.append(conns[:, i+1])

        ver_idx += 1
        if ver_idx == net_size:
            ver_idx = 0
            hor_idx += 1

    hor_idx = 0
    ver_idx = 0

    for i in range(0, hors.shape[1], 2):
        if hor_idx == 0 and ver_idx >= 3:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])
        elif hor_idx == 1 and ver_idx >= 2:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])
        elif hor_idx == 2 and ver_idx >= 1:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])
            if ver_idx == 3:
                ver_idx += 1
        elif hor_idx == 3 and ver_idx >= 1:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])

        ver_idx += 1
        if ver_idx == net_size + 1:
            ver_idx = 0
            hor_idx += 1

    hor_idx = 0
    ver_idx = 0

    for i in range(0, vers.shape[1], 2):
        if ver_idx == 0 and hor_idx >= 3:
            near_springs_outputs.append(vers[:, i])
            near_springs_outputs.append(vers[:, i+1])
        elif ver_idx == 1 and hor_idx >= 2:
            near_springs_outputs.append(vers[:, i])
            near_springs_outputs.append(vers[:, i+1])
        elif ver_idx == 2 and hor_idx >= 1:
            near_springs_outputs.append(vers[:, i])
            near_springs_outputs.append(vers[:, i+1])
        elif ver_idx == 3 and hor_idx >= 1:
            near_springs_outputs.append(vers[:, i])
            near_springs_outputs.append(vers[:, i+1])
        
        hor_idx += 1
        if hor_idx == net_size + 1:
            hor_idx = 0
            ver_idx += 1

    near_springs_outputs_arr = np.hstack([arr[:, np.newaxis] for arr in near_springs_outputs])

    return near_springs_outputs_arr

def near_actuation_selection_4(conns, hors, vers, net_size):
    near_actuation_outputs = []

    hor_idx = 0
    ver_idx = 0

    near_actuation_outputs = []

    hor_idx = 0
    ver_idx = 0

    for i in range(0, conns.shape[1], 2):
        if (hor_idx >= 1 and hor_idx <= 2) and (ver_idx >= 1 and ver_idx <= 2):
            near_actuation_outputs.append(conns[:, i])
            near_actuation_outputs.append(conns[:, i+1])

        ver_idx += 1
        if ver_idx == net_size:
            ver_idx = 0
            hor_idx += 1

    hor_idx = 0
    ver_idx = 0

    for i in range(0, hors.shape[1], 2):
        if (hor_idx >= 1 and hor_idx <= 2) and (ver_idx >= 1 and ver_idx <= 3):
            near_actuation_outputs.append(hors[:, i])
            near_actuation_outputs.append(hors[:, i+1])
            if hor_idx == 2 and ver_idx == 2:
                ver_idx += 1

        ver_idx += 1
        if ver_idx == net_size + 1:
            ver_idx = 0
            hor_idx += 1

    hor_idx = 0
    ver_idx = 0

    for i in range(0, vers.shape[1], 2):
        if (ver_idx >= 1 and ver_idx <= 2) and (hor_idx >= 1 and hor_idx <= 3):
            near_actuation_outputs.append(vers[:, i])
            near_actuation_outputs.append(vers[:, i+1])
        
        hor_idx += 1
        if hor_idx == net_size + 1:
            hor_idx = 0
            ver_idx += 1

    near_actuation_outputs_arr = np.hstack([arr[:, np.newaxis] for arr in near_actuation_outputs])

    return near_actuation_outputs_arr

if __name__ == "__main__":
    net_size = 6 #4 #
    stimulation_point_label = 69 #34 #
    vid_path = os.path.join(os.path.dirname(__file__), f"{net_size}by{net_size}/Videos")
    save_path = "C:/D/ABHYAAS/OneDrive - Virginia Tech/naughtonlab - active_projects/2025-JIMSS_paper/pdf_v4/Fig 4"
    
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

    # groups_dict = {
    #     'All Outputs': [],
    #     'Hors X Hors Y': [],
    #     'Hors X Vers X': [],
    #     'Hors X Vers Y': [],
    #     'Hors Y Vers X': [],
    #     'Hors Y Vers Y': [],
    #     'Vers X Vers Y': [],
    #     'Near Springs': [],
    #     'Near Actuation': []}

    groups_dict = {
        'All Outputs': [],
        'Hors Y Vers X': [],
        'Hors X Hors Y': [],
        'Hors X Vers Y': [],
        'Near Springs': [],
        'Near Actuation': []}
    
    labels_list = list(groups_dict.keys())
 
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
        groups_dict['All Outputs'].append((onc_train, onc_test, omc_train, omc_test))
        # print("All Outputs Done")

        output_conns = output_data[:, :conns_end]
        output_hors = output_data[:, conns_end:hors_end]
        output_hx = output_hors[:, ::2]
        output_hy = output_hors[:, 1::2]
        output_vers = output_data[:, hors_end:vers_end]
        output_vx = output_vers[:, ::2]
        output_vy = output_vers[:, 1::2]

        # Hors X Hors Y
        output_hxy = np.hstack((output_hx, output_hy))
        leg_cap_train_hxy, leg_cap_test_hxy, leg_R2_train_hxy, leg_R2_test_hxy, leg_rmse_train_hxy, leg_rmse_test_hxy = nonlinearity_testing(input_signal, output_hxy, leg_max_order, test_size, alpha_leg, regressor)
        mem_cap_train_hxy, mem_cap_test_hxy, mem_R2_train_hxy, mem_R2_test_hxy, mem_rmse_train_hxy, mem_rmse_test_hxy = memory_testing(input_signal, output_hxy, max_timesteps_back, test_size, alpha_mem, regressor)
        onc_train_hxy = sum(leg_cap_train_hxy)/len(leg_cap_train_hxy)
        onc_test_hxy = sum(leg_cap_test_hxy)/len(leg_cap_test_hxy)
        omc_train_hxy = sum(mem_cap_train_hxy)/len(mem_cap_train_hxy)
        omc_test_hxy = sum(mem_cap_test_hxy)/len(mem_cap_test_hxy)
        groups_dict['Hors X Hors Y'].append((onc_train_hxy, onc_test_hxy, omc_train_hxy, omc_test_hxy))
        # print("Hors X Hors Y Done")

        # Hors X Vers X
        # output_hxvx = np.hstack((output_hx, output_vx))
        # leg_cap_train_hxvx, leg_cap_test_hxvx, leg_R2_train_hxvx, leg_R2_test_hxvx, leg_rmse_train_hxvx, leg_rmse_test_hxvx = nonlinearity_testing(input_signal, output_hxvx, leg_max_order, test_size, alpha_leg, regressor)
        # mem_cap_train_hxvx, mem_cap_test_hxvx, mem_R2_train_hxvx, mem_R2_test_hxvx, mem_rmse_train_hxvx, mem_rmse_test_hxvx = memory_testing(input_signal, output_hxvx, max_timesteps_back, test_size, alpha_mem, regressor)
        # onc_train_hxvx = sum(leg_cap_train_hxvx)/len(leg_cap_train_hxvx)
        # onc_test_hxvx = sum(leg_cap_test_hxvx)/len(leg_cap_test_hxvx)
        # omc_train_hxvx = sum(mem_cap_train_hxvx)/len(mem_cap_train_hxvx)
        # omc_test_hxvx = sum(mem_cap_test_hxvx)/len(mem_cap_test_hxvx)
        # groups_dict['Hors X Vers X'].append((onc_train_hxvx, onc_test_hxvx, omc_train_hxvx, omc_test_hxvx))
        # print("Hors X Vers X Done")

        # Hors X Vers Y
        output_hxvy = np.hstack((output_hx, output_vy))
        leg_cap_train_hxvy, leg_cap_test_hxvy, leg_R2_train_hxvy, leg_R2_test_hxvy, leg_rmse_train_hxvy, leg_rmse_test_hxvy = nonlinearity_testing(input_signal, output_hxvy, leg_max_order, test_size, alpha_leg, regressor)
        mem_cap_train_hxvy, mem_cap_test_hxvy, mem_R2_train_hxvy, mem_R2_test_hxvy, mem_rmse_train_hxvy, mem_rmse_test_hxvy = memory_testing(input_signal, output_hxvy, max_timesteps_back, test_size, alpha_mem, regressor)
        onc_train_hxvy = sum(leg_cap_train_hxvy)/len(leg_cap_train_hxvy)
        onc_test_hxvy = sum(leg_cap_test_hxvy)/len(leg_cap_test_hxvy)
        omc_train_hxvy = sum(mem_cap_train_hxvy)/len(mem_cap_train_hxvy)
        omc_test_hxvy = sum(mem_cap_test_hxvy)/len(mem_cap_test_hxvy)
        groups_dict['Hors X Vers Y'].append((onc_train_hxvy, onc_test_hxvy, omc_train_hxvy, omc_test_hxvy))
        # print("Hors X Vers Y Done")

        # Hors Y Vers X
        output_hyvx = np.hstack((output_hy, output_vx))
        leg_cap_train_hyvx, leg_cap_test_hyvx, leg_R2_train_hyvx, leg_R2_test_hyvx, leg_rmse_train_hyvx, leg_rmse_test_hyvx = nonlinearity_testing(input_signal, output_hyvx, leg_max_order, test_size, alpha_leg, regressor)
        mem_cap_train_hyvx, mem_cap_test_hyvx, mem_R2_train_hyvx, mem_R2_test_hyvx, mem_rmse_train_hyvx, mem_rmse_test_hyvx = memory_testing(input_signal, output_hyvx, max_timesteps_back, test_size, alpha_mem, regressor)
        onc_train_hyvx = sum(leg_cap_train_hyvx)/len(leg_cap_train_hyvx)
        onc_test_hyvx = sum(leg_cap_test_hyvx)/len(leg_cap_test_hyvx)
        omc_train_hyvx = sum(mem_cap_train_hyvx)/len(mem_cap_train_hyvx)
        omc_test_hyvx = sum(mem_cap_test_hyvx)/len(mem_cap_test_hyvx)
        groups_dict['Hors Y Vers X'].append((onc_train_hyvx, onc_test_hyvx, omc_train_hyvx, omc_test_hyvx))
        # print("Hors Y Vers X Done")

        # Hors Y Vers Y
        # output_hyvy = np.hstack((output_hy, output_vy))
        # leg_cap_train_hyvy, leg_cap_test_hyvy, leg_R2_train_hyvy, leg_R2_test_hyvy, leg_rmse_train_hyvy, leg_rmse_test_hyvy = nonlinearity_testing(input_signal, output_hyvy, leg_max_order, test_size, alpha_leg, regressor)
        # mem_cap_train_hyvy, mem_cap_test_hyvy, mem_R2_train_hyvy, mem_R2_test_hyvy, mem_rmse_train_hyvy, mem_rmse_test_hyvy = memory_testing(input_signal, output_hyvy, max_timesteps_back, test_size, alpha_mem, regressor)
        # onc_train_hyvy = sum(leg_cap_train_hyvy)/len(leg_cap_train_hyvy)
        # onc_test_hyvy = sum(leg_cap_test_hyvy)/len(leg_cap_test_hyvy)
        # omc_train_hyvy = sum(mem_cap_train_hyvy)/len(mem_cap_train_hyvy)
        # omc_test_hyvy = sum(mem_cap_test_hyvy)/len(mem_cap_test_hyvy)
        # groups_dict['Hors Y Vers Y'].append((onc_train_hyvy, onc_test_hyvy, omc_train_hyvy, omc_test_hyvy))
        # print("Hors Y Vers Y Done")

        # Vers X Vers Y
        # output_vxvy = np.hstack((output_vx, output_vy))
        # leg_cap_train_vxvy, leg_cap_test_vxvy, leg_R2_train_vxvy, leg_R2_test_vxvy, leg_rmse_train_vxvy, leg_rmse_test_vxvy = nonlinearity_testing(input_signal, output_vxvy, leg_max_order, test_size, alpha_leg, regressor)
        # mem_cap_train_vxvy, mem_cap_test_vxvy, mem_R2_train_vxvy, mem_R2_test_vxvy, mem_rmse_train_vxvy, mem_rmse_test_vxvy = memory_testing(input_signal, output_vxvy, max_timesteps_back, test_size, alpha_mem, regressor)
        # onc_train_vxvy = sum(leg_cap_train_vxvy)/len(leg_cap_train_vxvy)
        # onc_test_vxvy = sum(leg_cap_test_vxvy)/len(leg_cap_test_vxvy)
        # omc_train_vxvy = sum(mem_cap_train_vxvy)/len(mem_cap_train_vxvy)
        # omc_test_vxvy = sum(mem_cap_test_vxvy)/len(mem_cap_test_vxvy)
        # groups_dict['Vers X Vers Y'].append((onc_train_vxvy, onc_test_vxvy, omc_train_vxvy, omc_test_vxvy))
        # print("Vers X Vers Y Done")

        # Near Springs
        if net_size == 6:
            output_near_springs = near_springs_selection_6(output_conns, output_hors, output_vers, net_size)
        elif net_size == 4:
            output_near_springs = near_springs_selection_4(output_conns, output_hors, output_vers, net_size)
        # print(output_near_springs.shape)
        leg_cap_train_ns, leg_cap_test_ns, leg_R2_train_ns, leg_R2_test_ns, leg_rmse_train_ns, leg_rmse_test_ns = nonlinearity_testing(input_signal, output_near_springs, leg_max_order, test_size, alpha_leg, regressor)
        mem_cap_train_ns, mem_cap_test_ns, mem_R2_train_ns, mem_R2_test_ns, mem_rmse_train_ns, mem_rmse_test_ns = memory_testing(input_signal, output_near_springs, max_timesteps_back, test_size, alpha_mem, regressor)
        onc_train_ns = sum(leg_cap_train_ns)/len(leg_cap_train_ns)
        onc_test_ns = sum(leg_cap_test_ns)/len(leg_cap_test_ns)
        omc_train_ns = sum(mem_cap_train_ns)/len(mem_cap_train_ns)
        omc_test_ns = sum(mem_cap_test_ns)/len(mem_cap_test_ns)
        groups_dict['Near Springs'].append((onc_train_ns, onc_test_ns, omc_train_ns, omc_test_ns))
        # print("Near Springs Done")

        # Near Actuation
        if net_size == 6:
            output_near_actuation = near_actuation_selection_6(output_conns, output_hors, output_vers, net_size)
        elif net_size == 4:
            output_near_actuation = near_actuation_selection_4(output_conns, output_hors, output_vers, net_size)
        # print(output_near_actuation.shape)
        leg_cap_train_na, leg_cap_test_na, leg_R2_train_na, leg_R2_test_na, leg_rmse_train_na, leg_rmse_test_na = nonlinearity_testing(input_signal, output_near_actuation, leg_max_order, test_size, alpha_leg, regressor)
        mem_cap_train_na, mem_cap_test_na, mem_R2_train_na, mem_R2_test_na, mem_rmse_train_na, mem_rmse_test_na = memory_testing(input_signal, output_near_actuation, max_timesteps_back, test_size, alpha_mem, regressor)
        onc_train_na = sum(leg_cap_train_na)/len(leg_cap_train_na)
        onc_test_na = sum(leg_cap_test_na)/len(leg_cap_test_na)
        omc_train_na = sum(mem_cap_train_na)/len(mem_cap_train_na)
        omc_test_na = sum(mem_cap_test_na)/len(mem_cap_test_na)
        groups_dict['Near Actuation'].append((onc_train_na, onc_test_na, omc_train_na, omc_test_na))
        # print("Near Actuation Done")

        # onc_train_list = [groups_dict[label][-1][0] for label in labels_list[1:]]
        # omc_train_list = [groups_dict[label][-1][2] for label in labels_list[1:]]
        # onc_test_list = [groups_dict[label][-1][1] for label in labels_list[1:]]
        # omc_test_list = [groups_dict[label][-1][3] for label in labels_list[1:]]

        # max_onc_train = max(onc_train_list)
        # max_omc_train = max(omc_train_list)
        # max_onc_test = max(onc_test_list)
        # max_omc_test = max(omc_test_list)
        # max_onc_train_label = onc_train_list.index(max_onc_train) + 1
        # max_omc_train_label = omc_train_list.index(max_omc_train) + 1
        # max_onc_test_label = onc_test_list.index(max_onc_test) + 1
        # max_omc_test_label = omc_test_list.index(max_omc_test) + 1
        # print(f"Angle: {angle}deg")
        # print(f"Max ONC Train: {labels_list[max_onc_train_label]} with Capacity: {max_onc_train}")
        # print(f"Max OMC Train: {labels_list[max_omc_train_label]} with Capacity: {max_omc_train}")
        # print(f"Max ONC Test: {labels_list[max_onc_test_label]} with Capacity: {max_onc_test}")
        # print(f"Max OMC Test: {labels_list[max_omc_test_label]} with Capacity: {max_omc_test}")
        # print("--------------------------")

        # plt.figure(figsize=(7.5, 7.5))
        # plt.plot(leg_x, legendre_capacity_test, '-o', label='All Outputs')
        # plt.plot(leg_x, leg_cap_test_hxy, '-o', label='Hors X Hors Y')
        # plt.plot(leg_x, leg_cap_test_hxvx, '-o', label='Hors X Vers X')
        # plt.plot(leg_x, leg_cap_test_hxvy, '-o', label='Hors X Vers Y')
        # plt.plot(leg_x, leg_cap_test_hyvx, '-o', label='Hors Y Vers X')
        # plt.plot(leg_x, leg_cap_test_hyvy, '-o', label='Hors Y Vers Y')
        # plt.plot(leg_x, leg_cap_test_vxvy, '-o', label='Vers X Vers Y')
        # plt.plot(leg_x, leg_cap_test_ns, '-o', label='Near Springs')
        # plt.plot(leg_x, leg_cap_test_na, '-o', label='Near Actuation')
        # plt.xlabel("Legendre Polynomial Order")
        # plt.ylabel("Capacity")
        # plt.title(f"Legendre Curves at Angle {angle}deg")
        # plt.legend()
        # plt.grid(True)
        # plt.ylim(-0.1, 1.1)
        # plt.savefig(f"{data_path}/feature_selection_legendre.png")
        # plt.close()

        # plt.figure(figsize=(7.5, 7.5))
        # plt.plot(mem_x, memory_capacity_test, '-o', label='All Outputs')
        # plt.plot(mem_x, mem_cap_test_hxy, '-o', label='Hors X Hors Y')
        # plt.plot(mem_x, mem_cap_test_hxvx, '-o', label='Hors X Vers X')
        # plt.plot(mem_x, mem_cap_test_hxvy, '-o', label='Hors X Vers Y')
        # plt.plot(mem_x, mem_cap_test_hyvx, '-o', label='Hors Y Vers X')
        # plt.plot(mem_x, mem_cap_test_hyvy, '-o', label='Hors Y Vers Y')
        # plt.plot(mem_x, mem_cap_test_vxvy, '-o', label='Vers X Vers Y')
        # plt.plot(mem_x, mem_cap_test_ns, '-o', label='Near Springs')
        # plt.plot(mem_x, mem_cap_test_na, '-o', label='Near Actuation')
        # plt.xlabel("Time Delay (s)")
        # plt.ylabel("Capacity")
        # plt.title(f"Memory Curves at Angle {angle}deg")
        # plt.legend()
        # plt.grid(True)
        # plt.ylim(-0.1, 1.1)
        # plt.savefig(f"{data_path}/feature_selection_memory.png")
        # plt.close()

        # plt.figure(figsize=(7.5, 7.5))
        # plt.bar(labels_list, [groups_dict[label][-1][1]/onc_test for label in labels_list])
        # plt.xlabel("Output Groups")
        # plt.ylabel("Capacity")
        # plt.title(f"ONC for Different Output Groups at Angle {angle}deg")
        # plt.xticks(rotation=45, ha='right')
        # plt.ylim(-0.1, max(1.1, max(onc_test_list)/onc_test + 0.1))
        # plt.grid(True)
        # plt.savefig(f"{data_path}/feature_selection_ONC_bar.png")
        # plt.close()

        # plt.figure(figsize=(7.5, 7.5))
        # plt.bar(labels_list, [groups_dict[label][-1][3]/omc_test for label in labels_list])
        # plt.xlabel("Output Groups")
        # plt.ylabel("Capacity")
        # plt.title(f"OMC for Different Output Groups at Angle {angle}deg")
        # plt.xticks(rotation=45, ha='right')
        # plt.ylim(-0.1, max(1.1, max(omc_test_list)/omc_test + 0.1))
        # plt.grid(True)
        # plt.savefig(f"{data_path}/feature_selection_OMC_bar.png")
        # plt.close()

    plt.figure(figsize=(7.5, 7.5))
    for label in labels_list:
        legendre_test = [item[1] for item in groups_dict[label]]
        plt.plot(angle_list, legendre_test, marker='o', label=f'{label}', linestyle='-')
    plt.xlabel("Input Angle (deg)")
    plt.ylabel("Capacity")
    plt.title(f"{net_size}by{net_size} Nonlinearity on Output Reduction")
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.savefig(f"{save_path}/Nonlinearity_{net_size}by{net_size}_output_reduction_Sample{sample_freq}_Update{eval_freq}.pdf", dpi=300)
    plt.show()
    
    plt.figure(figsize=(7.5, 7.5))
    for label in labels_list:
        memory_test = [item[3] for item in groups_dict[label]]
        plt.plot(angle_list, memory_test, marker='o', label=f'{label}', linestyle='-')
    plt.xlabel("Input Angle (deg)")
    plt.ylabel("Capacity")
    plt.title(f"{net_size}by{net_size} Memory on Output Reduction")
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.savefig(f"{save_path}/Memory_{net_size}by{net_size}_output_reduction_Sample{sample_freq}_Update{eval_freq}.pdf", dpi=300)
    plt.show()

    # avg_onc_test = []
    # avg_omc_test = []
    # for label in labels_list:
    #     relative_onc_test_sum = 0
    #     relative_omc_test_sum = 0
    #     # for i in range(len(angle_list)):
    #     i = -1
    #     relative_onc_test_sum += groups_dict[label][i][1]/groups_dict['All Outputs'][i][1]
    #     # print(relative_onc_test_sum)
    #     relative_omc_test_sum += groups_dict[label][i][3]/groups_dict['All Outputs'][i][3]
    #     # print(relative_omc_test_sum)
    #     # avg_onc_test.append(relative_onc_test_sum/len(angle_list))
    #     # avg_omc_test.append(relative_omc_test_sum/len(angle_list))
    #     avg_onc_test.append(relative_onc_test_sum)
    #     avg_omc_test.append(relative_omc_test_sum)

    # plt.figure(figsize=(7.5, 7.5))
    # plt.bar(labels_list, avg_onc_test)
    # plt.xlabel("Output Groups")
    # plt.ylabel("Capacity")
    # plt.title(f"{net_size}by{net_size} Average Nonlinearity on Output Reduction")
    # plt.xticks(rotation=15)
    # plt.ylim(-0.1, max(1.1, max(avg_onc_test)+0.1))
    # plt.grid(True)
    # plt.savefig(f"{save_path}/Set1_50deg_Nonlinearity_{net_size}by{net_size}_output_reduction_Sample{sample_freq}_Update{eval_freq}.pdf", dpi=300)
    # plt.show()

    # plt.figure(figsize=(7.5, 7.5))
    # plt.bar(labels_list, avg_omc_test)
    # plt.xlabel("Output Groups")
    # plt.ylabel("Capacity")
    # plt.title(f"{net_size}by{net_size} Average Memory on Output Reduction")
    # plt.xticks(rotation=15)
    # plt.ylim(-0.1, max(1.1, max(avg_omc_test)+0.1))
    # plt.grid(True)
    # plt.savefig(f"{save_path}/Set1_50deg_Memory_{net_size}by{net_size}_output_reduction_Sample{sample_freq}_Update{eval_freq}.pdf", dpi=300)
    # plt.show()