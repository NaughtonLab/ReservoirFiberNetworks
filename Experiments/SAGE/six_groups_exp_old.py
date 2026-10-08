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

def nonlinearity_testing(input_signal, output, leg_max_order, test_size, alpha):

    clf = Ridge(alpha=alpha)
    # clf = LinearRegression()

    legendre_capacity_train_cx = []
    legendre_capacity_test_cx = []
    legendre_R2_train_cx = []
    legendre_R2_test_cx = []
    legendre_rmse_train_cx = []
    legendre_rmse_test_cx = []

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

        legendre_capacity_train_cx.append(capacity_train)
        legendre_R2_train_cx.append(R2_train)
        legendre_rmse_train_cx.append(RMSE_train)

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

        legendre_capacity_test_cx.append(capacity_test)
        legendre_R2_test_cx.append(R2_test)
        legendre_rmse_test_cx.append(RMSE_test)
        
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

    return legendre_capacity_train_cx, legendre_capacity_test_cx, legendre_R2_train_cx, legendre_R2_test_cx, legendre_rmse_train_cx, legendre_rmse_test_cx

def memory_testing(input_signal, output, max_time_back, test_size, alpha):
    clf = Ridge(alpha=alpha)
    # clf = LinearRegression()

    memory_capacity_train_cx = []
    memory_capacity_test_cx = []
    memory_R2_train_cx = []
    memory_R2_test_cx = []
    memory_rmse_train_cx = []
    memory_rmse_test_cx = []

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

        memory_capacity_train_cx.append(capacity_train)
        memory_R2_train_cx.append(R2_train)
        memory_rmse_train_cx.append(RMSE_train)

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

        memory_capacity_test_cx.append(capacity_test)
        memory_R2_test_cx.append(R2_test)
        memory_rmse_test_cx.append(RMSE_test)

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

    return memory_capacity_train_cx, memory_capacity_test_cx, memory_R2_train_cx, memory_R2_test_cx, memory_rmse_train_cx, memory_rmse_test_cx

if __name__ == "__main__":
    net_size = 4 #6 #
    stimulation_point_label = 34 #69 #
    vid_path = os.path.join(os.path.dirname(__file__), f"{net_size}by{net_size}/Videos")
    # save_path = "C:/D/ABHYAAS/OneDrive - Virginia Tech/naughtonlab - active_projects/2025-JIMSS_paper/pdf_v3/Fig 4"
    
    if net_size == 4:
        c_labels = [i for i in range(5, 11+2, 2)] + [i for i in range(18, 24+2, 2)] + [i for i in range(31, 37+2, 2)] + [i for i in range(44, 50+2, 2)]
        h_labels = [i for i in range(4, 12+2, 2)] + [i for i in range(17, 25+2, 2)] + [i for i in range(30, 38+2, 2) if i != 34] + [i for i in range(43, 51+2, 2)]
        v_labels = [i for i in range(0, 3+1)] + [i for i in range(13, 16+1)] + [i for i in range(26, 29+1)] + [i for i in range(39, 42+1)] + [i for i in range(52, 55+1)]
        c_labels = [c if c <= 34 else c-1 for c in c_labels]
        h_labels = [h if h <= 34 else h-1 for h in h_labels]
        v_labels = [v if v <= 34 else v-1 for v in v_labels]
    elif net_size == 6:
        c_labels = [i for i in range(7, 17+2, 2)] + [i for i in range(26, 36+2, 2)] + [i for i in range(45, 55+2, 2)] + [i for i in range(64, 74+2, 2)] + [i for i in range(83, 93+2, 2)] + [i for i in range(102, 112+2, 2)]
        h_labels = [i for i in range(6, 18+2, 2)] + [i for i in range(25, 37+2, 2)] + [i for i in range(44, 56+2, 2)] + [i for i in range(63, 75+2, 2) if i != 69] + [i for i in range(82, 94+2, 2)] + [i for i in range(101, 113+2, 2)]
        v_labels = [i for i in range(0, 5+1)] + [i for i in range(19, 24+1)] + [i for i in range(38, 43+1)] + [i for i in range(57, 62+1)] + [i for i in range(76, 81+1)] + [i for i in range(95, 100+1)] + [i for i in range(114, 119+1)]
        c_labels = [c if c <= 69 else c-1 for c in c_labels]
        h_labels = [h if h <= 69 else h-1 for h in h_labels]
        v_labels = [v if v <= 69 else v-1 for v in v_labels]
    
    cx_labels = [c for c in c_labels]
    cy_labels = [c+1 for c in c_labels]
    hx_labels = [h for h in h_labels]
    hy_labels = [h+1 for h in h_labels]
    vx_labels = [v for v in v_labels]
    vy_labels = [v+1 for v in v_labels]

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

    leg_x = np.linspace(1, leg_max_order, leg_max_order)
    max_timesteps_back = np.rint(max_time_back_seconds * fps/frame_rate_for_img).astype(int)
    mem_x = np.linspace(0, max_time_back_seconds, max_timesteps_back+1)

    # groups_dict = {
    #     'All Outputs': [],
    #     'Conns X': [],
    #     'Conns Y': [],
    #     'Hors X': [],
    #     'Hors Y': [],
    #     'Vers X': [],
    #     'Vers Y': []}

    groups_dict = {
        'All Outputs': [],
        'Concatenated': []}
    
    for i in range(len(angle_list)):
        angle = angle_list[i]

        name = f"{sample_freq}_{eval_freq}_{angle}.MP4"

        data_path = f"{vid_path}/{name}_frames"

        # hors_xy = pd.read_csv(f'{data_path}/hors_xy.csv')
        # label_list_hxy = hors_xy['Label'].unique().tolist()
        # min_label_hxy = min(label_list_hxy)
        # max_label_hxy = max(label_list_hxy)
        # min_shape_hxy = np.min([len(hors_xy[hors_xy['Label'] == i]) for i in label_list_hxy])
        # vers_xy = pd.read_csv(f'{data_path}/vers_xy.csv')
        # label_list_vxy = vers_xy['Label'].unique().tolist()
        # min_label_vxy = min(label_list_vxy)
        # max_label_vxy = max(label_list_vxy)
        # min_shape_vxy = np.min([len(vers_xy[vers_xy['Label'] == i]) for i in label_list_vxy])
        # conns_xy = pd.read_csv(f'{data_path}/conns_xy.csv')
        # label_list_cxy = conns_xy['Label'].unique().tolist()
        # min_label_cxy = min(label_list_cxy)
        # max_label_cxy = max(label_list_cxy)
        # min_shape_cxy = np.min([len(conns_xy[conns_xy['Label'] == i]) for i in label_list_cxy])

        # clean_data = pd.read_csv(f'{data_path}/clean_data_final.csv')
        # clean_data['Time_s'] = clean_data['Time']/120
        # min_label = clean_data['Label'].min()
        # max_label = clean_data['Label'].max()

        # stimulation_point = clean_data[clean_data['Label'] == stimulation_point_label]
        # min_shape = np.min([len(clean_data[clean_data['Label'] == i]) for i in range(min_label, max_label+1)])
        # min_shape = np.min([min_shape_hxy, min_shape_vxy, min_shape_cxy])
        # print(min_shape)

        orig = np.load(f'{data_path}/eval_data.npz', allow_pickle=True)
        input_signal = orig['input_data']
        output_data = orig['output_data']
        max_labels = output_data.shape[1]
        min_shape = input_signal.shape[0]

        # Nonlinearity Testing
        legendre_capacity_train, legendre_capacity_test, legendre_R2_train, legendre_R2_test, legendre_rmse_train, legendre_rmse_test = nonlinearity_testing(input_signal, output_data, leg_max_order, test_size, alpha_leg)
        # Memory Testing
        memory_capacity_train, memory_capacity_test, memory_R2_train, memory_R2_test, memory_rmse_train, memory_rmse_test = memory_testing(input_signal, output_data, max_timesteps_back, test_size, alpha_mem)

        onc_train = sum(legendre_capacity_train)/len(legendre_capacity_train)
        onc_test = sum(legendre_capacity_test)/len(legendre_capacity_test)
        omc_train = sum(memory_capacity_train)/len(memory_capacity_train)
        omc_test = sum(memory_capacity_test)/len(memory_capacity_test)
        print("All Outputs Done")

        output_cx = output_data[:, cx_labels]
        output_cy = output_data[:, cy_labels]
        output_hx = output_data[:, hx_labels]
        output_hy = output_data[:, hy_labels]
        output_vx = output_data[:, vx_labels]
        output_vy = output_data[:, vy_labels]

        output_concat = np.hstack([output_cx, output_cy, output_hx, output_hy, output_vx, output_vy])
        print(output_data.shape, output_concat.shape)  

        # Nonlinearity testing
        legendre_capacity_train_concat, legendre_capacity_test_concat, legendre_R2_train_concat, legendre_R2_test_concat, legendre_rmse_train_concat, legendre_rmse_test_concat = nonlinearity_testing(input_signal, output_concat, leg_max_order, test_size, alpha_leg)
        # Memory testing
        memory_capacity_train_concat, memory_capacity_test_concat, memory_R2_train_concat, memory_R2_test_concat, memory_rmse_train_concat, memory_rmse_test_concat = memory_testing(input_signal, output_concat, max_timesteps_back, test_size, alpha_mem)

        onc_train_concat = sum(legendre_capacity_train_concat)/len(legendre_capacity_train_concat)
        onc_test_concat = sum(legendre_capacity_test_concat)/len(legendre_capacity_test_concat)
        omc_train_concat = sum(memory_capacity_train_concat)/len(memory_capacity_train_concat)
        omc_test_concat = sum(memory_capacity_test_concat)/len(memory_capacity_test_concat)          

        # "Conns X"
        # output_cx = None
        # for j in range(min_label_cxy, max_label_cxy+1):
        #     if j != stimulation_point_label and j in label_list_cxy:
        #         one_over_time = conns_xy[conns_xy['Label']==j]
        #         if one_over_time.empty:
        #             continue
        #         one_over_time = one_over_time.iloc[:, 5:6] - one_over_time.iloc[0, 5:6]
        #         array = one_over_time.to_numpy()
        #         array = array[:min_shape, :]
        #         if output_cx is None:
        #             output_cx = array
        #         else:
        #             output_cx = np.hstack([output_cx, array])     

        # "Conns Y"
        # output_cy = None
        # for j in range(min_label_cxy, max_label_cxy+1):
        #     if j != stimulation_point_label and j in label_list_cxy:
        #         one_over_time = conns_xy[conns_xy['Label']==j]
        #         if one_over_time.empty:
        #             continue
        #         one_over_time = one_over_time.iloc[:, 6:7] - one_over_time.iloc[0, 6:7]
        #         array = one_over_time.to_numpy()
        #         array = array[:min_shape, :]
        #         if output_cy is None:
        #             output_cy = array
        #         else:
        #             output_cy = np.hstack([output_cy, array])

        # "Hors X"
        # output_hx = None
        # for j in range(min_label_hxy, max_label_hxy+1):
        #     if j != stimulation_point_label and j in label_list_hxy:
        #         one_over_time = hors_xy[hors_xy['Label']==j]
        #         if one_over_time.empty:
        #             continue
        #         one_over_time = one_over_time.iloc[:, 5:6] - one_over_time.iloc[0, 5:6]
        #         array = one_over_time.to_numpy()
        #         array = array[:min_shape, :]
        #         if output_hx is None:
        #             output_hx = array
        #         else:
        #             output_hx = np.hstack([output_hx, array])

        # "Hors Y"
        # output_hy = None
        # for j in range(min_label_hxy, max_label_hxy+1):
        #     if j != stimulation_point_label and j in label_list_hxy:
        #         one_over_time = hors_xy[hors_xy['Label']==j]
        #         if one_over_time.empty:
        #             continue
        #         one_over_time = one_over_time.iloc[:, 6:7] - one_over_time.iloc[0, 6:7]
        #         array = one_over_time.to_numpy()
        #         array = array[:min_shape, :]
        #         if output_hy is None:
        #             output_hy = array
        #         else:
        #             output_hy = np.hstack([output_hy, array])

        # "Vers X"
        # output_vx = None
        # for j in range(min_label_vxy, max_label_vxy+1):
        #     if j != stimulation_point_label and j in label_list_vxy:
        #         one_over_time = vers_xy[vers_xy['Label']==j]
        #         if one_over_time.empty:
        #             continue
        #         one_over_time = one_over_time.iloc[:, 5:6] - one_over_time.iloc[0, 5:6]
        #         array = one_over_time.to_numpy()
        #         array = array[:min_shape, :]
        #         if output_vx is None:
        #             output_vx = array
        #         else:
        #             output_vx = np.hstack([output_vx, array])

        # "Vers Y"
        # output_vy = None
        # for j in range(min_label_vxy, max_label_vxy+1):
        #     if j != stimulation_point_label and j in label_list_vxy:
        #         one_over_time = vers_xy[vers_xy['Label']==j]
        #         if one_over_time.empty:
        #             continue
        #         one_over_time = one_over_time.iloc[:, 6:7] - one_over_time.iloc[0, 6:7]
        #         array = one_over_time.to_numpy()
        #         array = array[:min_shape, :]
        #         if output_vy is None:
        #             output_vy = array
        #         else:
        #             output_vy = np.hstack([output_vy, array])
        
        # output_cx /= np.mean(output_data, axis=0)
        # output_cx /= np.std(output_data, axis=0)
        # scaler = preprocessing.StandardScaler()
        # output_cx = scaler.fit_transform(output_cx)

        # output_cx = output_cx[transience:min_shape, :]

        # # Nonlinearity testing
        # legendre_capacity_train_cx, legendre_capacity_test_cx, legendre_R2_train_cx, legendre_R2_test_cx, legendre_rmse_train_cx, legendre_rmse_test_cx = nonlinearity_testing(input_signal, output_cx, leg_max_order, test_size, alpha_leg)

        # # Memory testing
        # memory_capacity_train_cx, memory_capacity_test_cx, memory_R2_train_cx, memory_R2_test_cx, memory_rmse_train_cx, memory_rmse_test_cx = memory_testing(input_signal, output_cx, max_timesteps_back, test_size, alpha_mem)

        # onc_train_cx = sum(legendre_capacity_train_cx)/len(legendre_capacity_train_cx)
        # onc_test_cx = sum(legendre_capacity_test_cx)/len(legendre_capacity_test_cx)
        # omc_train_cx = sum(memory_capacity_train_cx)/len(memory_capacity_train_cx)
        # omc_test_cx = sum(memory_capacity_test_cx)/len(memory_capacity_test_cx)

        # print("Conns X Done")
        
        

        # output_cy /= np.mean(output_cy, axis=0)
        # output_cy /= np.std(output_cy, axis=0)
        # scaler = preprocessing.StandardScaler()
        # output_cy = scaler.fit_transform(output_cy)
        # output_cy = output_cy[transience:min_shape, :]

        # # Nonlinearity testing
        # legendre_capacity_train_cy, legendre_capacity_test_cy, legendre_R2_train_cy, legendre_R2_test_cy, legendre_rmse_train_cy, legendre_rmse_test_cy = nonlinearity_testing(input_signal, output_cy, leg_max_order, test_size, alpha_leg)
        # # Memory testing
        # memory_capacity_train_cy, memory_capacity_test_cy, memory_R2_train_cy, memory_R2_test_cy, memory_rmse_train_cy, memory_rmse_test_cy = memory_testing(input_signal, output_cy, max_timesteps_back, test_size, alpha_mem)
        
        # onc_train_cy = sum(legendre_capacity_train_cy)/len(legendre_capacity_train_cy)
        # onc_test_cy = sum(legendre_capacity_test_cy)/len(legendre_capacity_test_cy)
        # omc_train_cy = sum(memory_capacity_train_cy)/len(memory_capacity_train_cy)
        # omc_test_cy = sum(memory_capacity_test_cy)/len(memory_capacity_test_cy)

        # print("Conns Y Done")

        

        # output_hx /= np.mean(output_hx, axis=0)
        # output_hx /= np.std(output_hx, axis=0)
        # scaler = preprocessing.StandardScaler()
        # output_hx = scaler.fit_transform(output_hx)
        # output_hx = output_hx[transience:min_shape, :]

        # # Nonlinearity testing
        # legendre_capacity_train_hx, legendre_capacity_test_hx, legendre_R2_train_hx, legendre_R2_test_hx, legendre_rmse_train_hx, legendre_rmse_test_hx = nonlinearity_testing(input_signal, output_hx, leg_max_order, test_size, alpha_leg)
        # # Memory testing
        # memory_capacity_train_hx, memory_capacity_test_hx, memory_R2_train_hx, memory_R2_test_hx, memory_rmse_train_hx, memory_rmse_test_hx = memory_testing(input_signal, output_hx, max_timesteps_back, test_size, alpha_mem)
        
        # onc_train_hx = sum(legendre_capacity_train_hx)/len(legendre_capacity_train_hx)
        # onc_test_hx = sum(legendre_capacity_test_hx)/len(legendre_capacity_test_hx)
        # omc_train_hx = sum(memory_capacity_train_hx)/len(memory_capacity_train_hx)
        # omc_test_hx = sum(memory_capacity_test_hx)/len(memory_capacity_test_hx)

        # print("Hors X Done")

        

        # output_hy /= np.mean(output_hy, axis=0)
        # output_hy /= np.std(output_hy, axis=0)
        # scaler = preprocessing.StandardScaler()
        # output_hy = scaler.fit_transform(output_hy)
        # output_hy = output_hy[transience:min_shape, :]

        # # Nonlinearity testing
        # legendre_capacity_train_hy, legendre_capacity_test_hy, legendre_R2_train_hy, legendre_R2_test_hy, legendre_rmse_train_hy, legendre_rmse_test_hy = nonlinearity_testing(input_signal, output_hy, leg_max_order, test_size, alpha_leg)
        # # Memory testing
        # memory_capacity_train_hy, memory_capacity_test_hy, memory_R2_train_hy, memory_R2_test_hy, memory_rmse_train_hy, memory_rmse_test_hy = memory_testing(input_signal, output_hy, max_timesteps_back, test_size, alpha_mem)
        # onc_train_hy = sum(legendre_capacity_train_hy)/len(legendre_capacity_train_hy)
        # onc_test_hy = sum(legendre_capacity_test_hy)/len(legendre_capacity_test_hy)
        # omc_train_hy = sum(memory_capacity_train_hy)/len(memory_capacity_train_hy)
        # omc_test_hy = sum(memory_capacity_test_hy)/len(memory_capacity_test_hy)

        # print("Hors Y Done")

        
        
        # output_vx /= np.mean(output_vx, axis=0)
        # output_vx /= np.std(output_vx, axis=0)
        # scaler = preprocessing.StandardScaler()
        # output_vx = scaler.fit_transform(output_vx)
        # output_vx = output_vx[transience:min_shape, :]

        # # Nonlinearity testing
        # legendre_capacity_train_vx, legendre_capacity_test_vx, legendre_R2_train_vx, legendre_R2_test_vx, legendre_rmse_train_vx, legendre_rmse_test_vx = nonlinearity_testing(input_signal, output_vx, leg_max_order, test_size, alpha_leg)
        # # Memory testing
        # memory_capacity_train_vx, memory_capacity_test_vx, memory_R2_train_vx, memory_R2_test_vx, memory_rmse_train_vx, memory_rmse_test_vx = memory_testing(input_signal, output_vx, max_timesteps_back, test_size, alpha_mem)
        
        # onc_train_vx = sum(legendre_capacity_train_vx)/len(legendre_capacity_train_vx)
        # onc_test_vx = sum(legendre_capacity_test_vx)/len(legendre_capacity_test_vx)
        # omc_train_vx = sum(memory_capacity_train_vx)/len(memory_capacity_train_vx)
        # omc_test_vx = sum(memory_capacity_test_vx)/len(memory_capacity_test_vx)

        # print("Vers X Done")
        
        
        
        # output_vy /= np.mean(output_vy, axis=0)
        # output_vy /= np.std(output_vy, axis=0)
        # scaler = preprocessing.StandardScaler()
        # output_vy = scaler.fit_transform(output_vy)
        # output_vy = output_vy[transience:min_shape, :]
        
        # # Nonlinearity testing
        # legendre_capacity_train_vy, legendre_capacity_test_vy, legendre_R2_train_vy, legendre_R2_test_vy, legendre_rmse_train_vy, legendre_rmse_test_vy = nonlinearity_testing(input_signal, output_vy, leg_max_order, test_size, alpha_leg)
        # # Memory testing
        # memory_capacity_train_vy, memory_capacity_test_vy, memory_R2_train_vy, memory_R2_test_vy, memory_rmse_train_vy, memory_rmse_test_vy = memory_testing(input_signal, output_vy, max_timesteps_back, test_size, alpha_mem)
        
        # onc_train_vy = sum(legendre_capacity_train_vy)/len(legendre_capacity_train_vy)
        # onc_test_vy = sum(legendre_capacity_test_vy)/len(legendre_capacity_test_vy)
        # omc_train_vy = sum(memory_capacity_train_vy)/len(memory_capacity_train_vy)
        # omc_test_vy = sum(memory_capacity_test_vy)/len(memory_capacity_test_vy)

        # print("Vers Y Done")

        # onc_train_list = [onc_train_cx, onc_train_cy, onc_train_hx, onc_train_hy, onc_train_vx, onc_train_vy]
        # onc_test_list = [onc_test_cx, onc_test_cy, onc_test_hx, onc_test_hy, onc_test_vx, onc_test_vy]
        # omc_train_list = [omc_train_cx, omc_train_cy, omc_train_hx, omc_train_hy, omc_train_vx, omc_train_vy]
        # omc_test_list = [omc_test_cx, omc_test_cy, omc_test_hx, omc_test_hy, omc_test_vx, omc_test_vy]

        # max_onc_train = max(onc_train_list)
        # max_onc_test = max(onc_test_list)
        # max_omc_train = max(omc_train_list)
        # max_omc_test = max(omc_test_list)
        # idx_max_onc_train = onc_train_list.index(max_onc_train)
        # idx_max_onc_test = onc_test_list.index(max_onc_test)
        # idx_max_omc_train = omc_train_list.index(max_omc_train)
        # idx_max_omc_test = omc_test_list.index(max_omc_test)

        # labels_list = ['Conns X', 'Conns Y', 'Hors X', 'Hors Y', 'Vers X', 'Vers Y']

        # print(f"Angle: {angle} degrees")
        # print(f"Original OnC and OmC Results for Train and Test Sets: {[onc_train, onc_test, omc_train, omc_test]}")
        # print(f"Max OnC Train at: {labels_list[idx_max_onc_train]} with value {max_onc_train}")
        # print(f"Max OnC Test at: {labels_list[idx_max_onc_test]} with value {max_onc_test}")
        # print(f"Max OmC Train at: {labels_list[idx_max_omc_train]} with value {max_omc_train}")
        # print(f"Max OmC Test at: {labels_list[idx_max_omc_test]} with value {max_omc_test}")
        # print(f"OnC Train: {onc_train_list}")
        # print(f"OnC Test: {onc_test_list}")
        # print(f"OmC Train: {omc_train_list}")
        # print(f"OmC Test: {omc_test_list}")
        # print("--------------------------------------------------")

        # onc_train_list = [onc_train] + onc_train_list
        # onc_test_list = [onc_test] + onc_test_list
        # omc_train_list = [omc_train] + omc_train_list
        # omc_test_list = [omc_test] + omc_test_list
        # labels_list = ['All Outputs'] + labels_list

        # plt.figure(figsize=(7.5, 7.5))
        # plt.bar(labels_list, onc_train_list, label='Training OnC')
        # plt.bar(labels_list, onc_test_list, alpha=0.6, label='Testing OnC')
        # plt.ylabel("Overall Nonlinear Capacity (OnC)")
        # plt.title(f"Overall Nonlinear Capacity (OnC) at {angle} degrees")
        # plt.ylim(0, onc_train_list[0]+0.1)
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f"{data_path}/overall_nonlinear_capacity_at_{angle}_degrees.png")
        # # plt.show()
        # plt.close()
        # plt.figure(figsize=(7.5, 7.5))
        # plt.bar(labels_list, omc_train_list, label='Training OmC')
        # plt.bar(labels_list, omc_test_list, alpha=0.6, label='Testing OmC')
        # plt.ylabel("Overall Memory Capacity (OmC)")
        # plt.title(f"Overall Memory Capacity (OmC) at {angle} degrees")
        # plt.ylim(0, omc_train_list[0]+0.1)
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f"{data_path}/overall_memory_capacity_at_{angle}_degrees.png")
        # # plt.show()
        # plt.close()

        # groups_dict['All Outputs'].append([onc_test, omc_test])
        # groups_dict['Conns X'].append([onc_test_cx, omc_test_cx])
        # groups_dict['Conns Y'].append([onc_test_cy, omc_test_cy])
        # groups_dict['Hors X'].append([onc_test_hx, omc_test_hx])
        # groups_dict['Hors Y'].append([onc_test_hy, omc_test_hy])
        # groups_dict['Vers X'].append([onc_test_vx, omc_test_vx])
        # groups_dict['Vers Y'].append([onc_test_vy, omc_test_vy])


        plt.figure(figsize=(7.5, 7.5))
        plt.plot(leg_x, legendre_capacity_test, '-o', label='Orignal')
        plt.plot(leg_x, legendre_capacity_test_concat, '-o', label='Concatenated')
        plt.xlabel("Legendre Polynomial Order")
        plt.ylabel("Nonlinear Capacity")
        plt.title(f"Nonlinear Capacity vs Legendre Polynomial Order at {angle} degrees")
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{data_path}/ONC_comparison_with_concat.png")
        plt.show()
        plt.close()

        plt.figure(figsize=(7.5, 7.5))
        plt.plot(mem_x, memory_capacity_test, '-o', label='Orignal')
        plt.plot(mem_x, memory_capacity_test_concat, '-o', label='Concatenated')
        plt.xlabel("Time Back (seconds)")
        plt.ylabel("Memory Capacity")
        plt.title(f"Memory Capacity vs Time Back at {angle} degrees")
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{data_path}/OMC_comparison_with_concat.png")
        plt.show()
        plt.close()

        groups_dict['All Outputs'].append([onc_test, omc_test])
        groups_dict['Concatenated'].append([onc_test_concat, omc_test_concat])

    plt.figure(figsize=(7.5, 7.5))
    for key in groups_dict.keys():
        onc_values = [item[0] for item in groups_dict[key]]
        plt.plot(angle_list, onc_values, '-o', label=key)
    plt.xlabel("Stimulation Angle (degrees)")
    plt.ylabel("Overall Nonlinear Capacity (OnC)")
    plt.title("Comparison with Concatenated Outputs (ONC)")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{vid_path}/ONC_comparison_with_concat.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(7.5, 7.5))
    for key in groups_dict.keys():
        omc_values = [item[1] for item in groups_dict[key]]
        plt.plot(angle_list, omc_values, '-o', label=key)
    plt.xlabel("Stimulation Angle (degrees)")
    plt.ylabel("Overall Memory Capacity (OmC)")
    plt.title("Comparison with Concatenated Outputs (OMC)")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{vid_path}/OMC_comparison_with_concat.png")
    plt.show()
    plt.close()       

    
    # plt.figure(figsize=(7.5, 7.5))
    # for key in groups_dict.keys():
    #     onc_values = [item[0] for item in groups_dict[key]]
    #     plt.plot(angle_list, onc_values, '-o', label=key)
    # plt.xlabel("Stimulation Angle (degrees)")
    # plt.ylabel("Overall Nonlinear Capacity (OnC)")
    # plt.title("Overall Nonlinear Capacity (OnC) vs Stimulation Angle")
    # plt.ylim(-0.1, 1.1)
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"{vid_path}/overall_nonlinear_capacity_vs_stimulation_angle.png")
    # plt.show()
    # plt.close()
    # plt.figure(figsize=(7.5, 7.5))
    # for key in groups_dict.keys():
    #     omc_values = [item[1] for item in groups_dict[key]]
    #     plt.plot(angle_list, omc_values, '-o', label=key)
    # plt.xlabel("Stimulation Angle (degrees)")
    # plt.ylabel("Overall Memory Capacity (OmC)")
    # plt.title("Overall Memory Capacity (OmC) vs Stimulation Angle")
    # plt.ylim(-0.1, 1.1)
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"{vid_path}/overall_memory_capacity_vs_stimulation_angle.png")
    # plt.show()
    # plt.close()
