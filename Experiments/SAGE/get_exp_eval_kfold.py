import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from scipy.special import legendre

def nonlinearity_testing(input_signal, output, leg_max_order, test_size, alpha, regressor, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
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

        # x_train, x_test, y_train, y_test = train_test_split(x, y, 
        #                                                     test_size=test_size,
        #                                                     random_state=42,
        #                                                     shuffle=True)

        # Using KFold
        kf_capacity_train_sum = 0
        kf_capacity_test_sum = 0
        kf_r2_train_sum = 0
        kf_r2_test_sum = 0
        kf_rmse_train_sum = 0
        kf_rmse_test_sum = 0
        for i, (train_index, test_index) in enumerate(kf.split(x)):
            x_train, x_test, y_train, y_test = x[train_index, :], x[test_index, :], y[train_index, :], y[test_index, :]
    

            clf.fit(x_train, y_train)

            # Training capacity
            y_train_pred = clf.predict(x_train)

            idx = np.where(abs(y_train_pred) > 2)
            y_train_pred[idx] = np.mean(y)
            y_train[idx, 0] = np.mean(y)

            y2_train = (1/len(y_train)) * np.sum((y_train - np.mean(y_train))**2)

            MSE_train = mean_squared_error(y_train, y_train_pred)
            kf_capacity_train = 1 - MSE_train/y2_train
            kf_R2_train = r2_score(y_train, y_train_pred)
            kf_RMSE_train = root_mean_squared_error(y_train, y_train_pred)

            if kf_R2_train < 0:
                kf_R2_train = 0
            if kf_capacity_train < 0:
                kf_capacity_train = 0
            if kf_RMSE_train < 0:
                kf_RMSE_train = 0

            # Testing capacity
            y_test_pred = clf.predict(x_test)

            idx = np.where(abs(y_test_pred) > 2)
            y_test_pred[idx] = np.mean(y)
            y_test[idx, 0] = np.mean(y)

            y2_test = (1/len(y_test)) * np.sum((y_test - np.mean(y_test))**2)

            MSE_test = mean_squared_error(y_test, y_test_pred)
            kf_capacity_test = 1 - MSE_test/y2_test
            kf_R2_test = r2_score(y_test, y_test_pred)
            kf_RMSE_test = root_mean_squared_error(y_test, y_test_pred)

            if kf_R2_test < 0:
                kf_R2_test = 0
            if kf_capacity_test < 0:
                kf_capacity_test = 0
            if kf_RMSE_test < 0:
                kf_RMSE_test = 0

            kf_capacity_train_sum += kf_capacity_train
            kf_capacity_test_sum += kf_capacity_test
            kf_r2_train_sum += kf_R2_train
            kf_r2_test_sum += kf_R2_test
            kf_rmse_train_sum += kf_RMSE_train
            kf_rmse_test_sum += kf_RMSE_test

        capacity_train = kf_capacity_train_sum / n_splits
        capacity_test = kf_capacity_test_sum / n_splits
        R2_train = kf_r2_train_sum / n_splits
        R2_test = kf_r2_test_sum / n_splits
        RMSE_train = kf_rmse_train_sum / n_splits
        RMSE_test = kf_rmse_test_sum / n_splits

        legendre_capacity_train.append(capacity_train)
        legendre_R2_train.append(R2_train)
        legendre_rmse_train.append(RMSE_train)
        legendre_capacity_test.append(capacity_test)
        legendre_R2_test.append(R2_test)
        legendre_rmse_test.append(RMSE_test)
            

    return legendre_capacity_train, legendre_capacity_test, legendre_R2_train, legendre_R2_test, legendre_rmse_train, legendre_rmse_test

def memory_testing(input_signal, output, max_time_back, test_size, alpha, regressor, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=False)
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

        # x_train, x_test, y_train, y_test = train_test_split(x, y, 
        #                                                     test_size=test_size,
        #                                                     random_state=42,
        #                                                     shuffle=False)

        # Using KFold
        kf_capacity_train_sum = 0
        kf_capacity_test_sum = 0
        kf_r2_train_sum = 0
        kf_r2_test_sum = 0
        kf_rmse_train_sum = 0
        kf_rmse_test_sum = 0
        for i, (train_index, test_index) in enumerate(kf.split(x)):
            x_train, x_test, y_train, y_test = x[train_index, :], x[test_index, :], y[train_index, :], y[test_index, :]
            
            clf.fit(x_train, y_train)

            # Training capacity
            y_train_pred = clf.predict(x_train)

            idx = np.where(abs(y_train_pred) > 1.5)
            y_train_pred[idx] = np.mean(y)
            y_train[idx, 0] = np.mean(y)

            y2_train = (1/len(y_train)) * np.sum((y_train - np.mean(y_train))**2)

            MSE_train = mean_squared_error(y_train, y_train_pred)
            kf_capacity_train = 1 - MSE_train/y2_train
            kf_R2_train = r2_score(y_train, y_train_pred)
            kf_RMSE_train = root_mean_squared_error(y_train, y_train_pred)

            if kf_R2_train < 0:
                kf_R2_train = 0
            if kf_capacity_train < 0:
                kf_capacity_train = 0
            if kf_RMSE_train < 0:
                kf_RMSE_train = 0

            # Testing capacity
            y_test_pred = clf.predict(x_test)

            idx = np.where(abs(y_test_pred) > 1.5)
            y_test_pred[idx] = np.mean(y)
            y_test[idx, 0] = np.mean(y)

            y2_test = (1/len(y_test)) * np.sum((y_test - np.mean(y_test))**2)

            MSE_test = mean_squared_error(y_test, y_test_pred)
            kf_capacity_test = 1 - MSE_test/y2_test
            kf_R2_test = r2_score(y_test, y_test_pred)
            kf_RMSE_test = root_mean_squared_error(y_test, y_test_pred)

            if kf_R2_test < 0:
                kf_R2_test = 0
            if kf_capacity_test < 0:
                kf_capacity_test = 0
            if kf_RMSE_test < 0:
                kf_RMSE_test = 0

            kf_capacity_train_sum += kf_capacity_train
            kf_capacity_test_sum += kf_capacity_test
            kf_r2_train_sum += kf_R2_train
            kf_r2_test_sum += kf_R2_test
            kf_rmse_train_sum += kf_RMSE_train
            kf_rmse_test_sum += kf_RMSE_test
        
        capacity_train = kf_capacity_train_sum / n_splits
        capacity_test = kf_capacity_test_sum / n_splits
        R2_train = kf_r2_train_sum / n_splits
        R2_test = kf_r2_test_sum / n_splits
        RMSE_train = kf_rmse_train_sum / n_splits
        RMSE_test = kf_rmse_test_sum / n_splits

        memory_capacity_train.append(capacity_train)
        memory_R2_train.append(R2_train)
        memory_rmse_train.append(RMSE_train)
        memory_capacity_test.append(capacity_test)
        memory_R2_test.append(R2_test)
        memory_rmse_test.append(RMSE_test)

    return memory_capacity_train, memory_capacity_test, memory_R2_train, memory_R2_test, memory_rmse_train, memory_rmse_test

if __name__ == "__main__":
    # save_path = "C:/D/ABHYAAS/OneDrive - Virginia Tech/naughtonlab - active_projects/2025-JIMSS_paper/pdf_v3/Fig 4"
    
    net_size = 4 #6 #    
    stimulation_point_label = 34 #69 #
    vid_path = os.path.join(os.path.dirname(__file__), f"{net_size}by{net_size}/Videos")
    save_path = f"{vid_path}/corrected"

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
    n_splits = 10

    df = pd.DataFrame(columns=['sample_freq', 'eval_freq', 'angle', 'onc_train', 'omc_train', 'onc_test', 'omc_test'])

    for i in range(len(angle_list)):
        angle = angle_list[i]

        name = f"{sample_freq}_{eval_freq}_{angle}.MP4"

        data_path = f"{vid_path}/{name}_frames"

        data = np.load(f"{data_path}/eval_data_concat.npz")
        input_signal = data['input_data']
        output = data['output_data']       

        matplotlib.rc('pdf', fonttype=42)

        # Nonlinearity testing
        legendre_capacity_train, legendre_capacity_test, legendre_R2_train, legendre_R2_test, legendre_rmse_train, legendre_rmse_test = nonlinearity_testing(input_signal, output, leg_max_order, test_size, alpha_leg, regressor, n_splits)

        # Memory testing       
        memory_capacity_train, memory_capacity_test, memory_R2_train, memory_R2_test, memory_rmse_train, memory_rmse_test = memory_testing(input_signal, output, max_timesteps_back, test_size, alpha_mem, regressor, n_splits)
                
        print(f"Overall Nonlinearity Capacity: {sum(legendre_capacity_test)/len(legendre_capacity_test)}, Overall Memory Capacity: {sum(memory_capacity_test)/len(memory_capacity_test)}")

        np.savez(f"{data_path}/eval_data_{n_splits}kfold.npz",
                 legendre_capacity_train=legendre_capacity_train,
                 legendre_capacity_test=legendre_capacity_test,
                 memory_capacity_train=memory_capacity_train,
                 memory_capacity_test=memory_capacity_test,
                 legendre_R2_train=legendre_R2_train,
                 legendre_R2_test=legendre_R2_test,
                 memory_R2_train=memory_R2_train,
                 memory_R2_test=memory_R2_test,
                 legendre_rmse_train=legendre_rmse_train,
                 legendre_rmse_test=legendre_rmse_test,
                 memory_rmse_train=memory_rmse_train,
                 memory_rmse_test=memory_rmse_test)
        
        df.at[i, 'sample_freq'] = sample_freq
        df.at[i, 'eval_freq'] = eval_freq
        df.at[i, 'angle'] = angle
        df.at[i, 'onc_train'] = sum(legendre_capacity_train)/len(legendre_capacity_train)
        df.at[i, 'omc_train'] = sum(memory_capacity_train)/len(memory_capacity_train)
        df.at[i, 'onc_test'] = sum(legendre_capacity_test)/len(legendre_capacity_test)
        df.at[i, 'omc_test'] = sum(memory_capacity_test)/len(memory_capacity_test)

    df.to_csv(f"{vid_path}/grid_search_results_{n_splits}kfold_{sample_freq}_{eval_freq}_alpha_leg{alpha_leg}_alpha_mem{alpha_mem}.csv", index=False)

    # plt.figure(figsize=(7.5, 7.5))
    # plt.plot(angle_list, df['onc_test'], '-o', markersize = 10, linewidth = 2)
    # plt.xlabel("Angle of Stimulation (deg)")
    # plt.ylabel("Capacity")
    # plt.ylim(-0.1, 1.1)
    # plt.xticks(angle_list)
    # plt.title(f"Nonlinear Capacity: Sample Freq: {sample_freq} Hz, Update Freq: {eval_freq} Hz")
    # plt.savefig(f"{save_path}/Nonlinearity_{net_size}by{net_size}_grid_search_Sample{sample_freq}_Update{eval_freq}_{n_splits}kfold.pdf", dpi=300)
    # plt.close()

    # plt.figure(figsize=(7.5, 7.5))
    # plt.plot(angle_list, df['omc_test'], '-o', markersize = 10, linewidth = 2)
    # plt.xlabel("Angle of Stimulation (deg)")
    # plt.ylabel("Capacity")
    # plt.ylim(-0.1, 1.1)
    # plt.xticks(angle_list)
    # plt.title(f"Memory Capacity: Sample Freq: {sample_freq} Hz, Update Freq: {eval_freq} Hz")
    # plt.savefig(f"{save_path}/Memory_{net_size}by{net_size}_grid_search_Sample{sample_freq}_Update{eval_freq}_{n_splits}kfold.pdf", dpi=300)
    # plt.close()    
    # plt.show()