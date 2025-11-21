import pickle
import numpy as np
from scipy.special import legendre
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

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

    # Ignoring first 50 samples to avoid initial transients
    x = x[50:, :]
    y = y[50:, :]

    # y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)

    # plt.figure(figsize=(20, 5))
    # plt.subplot(211)
    # plt.plot(input)
    # plt.title('Input Signal')
    # plt.subplot(212)
    # plt.plot(y)
    # plt.title('NARMA-2 Target Signal')
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
    clf.fit(x_train, y_train)

    # Training
    y_train_pred = clf.predict(x_train)

    RMSE_train = root_mean_squared_error(y_true=y_train, y_pred=y_train_pred)
    # capacity_train = 1 - MSE/y2
    R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

    # Testing
    y_test_pred = clf.predict(x_test)

    RMSE_test = root_mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    # capacity_test = 1 - RMSE/y2
    R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

    if R2_test < 0:
        R2_test = 0
    if R2_train < 0:
        R2_train = 0

    return RMSE_train, RMSE_test, R2_train, R2_test, y_train, y_train_pred, y_test, y_test_pred

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

    # Ignoring first 50 samples to avoid initial transients
    x = x[50:, :]
    y = y[50:, :]

    # y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
    clf.fit(x_train, y_train)

    # Training
    y_train_pred = clf.predict(x_train)

    RMSE_train = root_mean_squared_error(y_true=y_train, y_pred=y_train_pred)
    # capacity_train = 1 - MSE/y2
    R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

    # Testing
    y_test_pred = clf.predict(x_test)

    RMSE_test = root_mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    # capacity_test = 1 - MSE/y2
    R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

    if R2_test < 0:
        R2_test = 0
    if R2_train < 0:
        R2_train = 0

    return RMSE_train, RMSE_test, R2_train, R2_test, y_train, y_train_pred, y_test, y_test_pred

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

    # Ignoring first 50 samples to avoid initial transients
    x = x[50:, :]
    y = y[50:, :]

    # y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
    clf.fit(x_train, y_train)

    # Training
    y_train_pred = clf.predict(x_train)

    RMSE_train = root_mean_squared_error(y_true=y_train, y_pred=y_train_pred)
    # capacity_train = 1 - MSE/y2
    R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

    # Testing
    y_test_pred = clf.predict(x_test)

    RMSE_test = root_mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    # capacity_test = 1 - MSE/y2
    R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

    if R2_test < 0:
        R2_test = 0
    if R2_train < 0:
        R2_train = 0

    return RMSE_train, RMSE_test, R2_train, R2_test, y_train, y_train_pred, y_test, y_test_pred

if __name__ == '__main__':
    # every frame = 250 sf YM 100
    dt_list = [0.004, 0.012, 0.02, 0.048, 0.1]

    narma_score_sum_array = np.zeros((len(dt_list), 6)) # dt, spec, train/test, narma2/5/10

    for j in range(len(dt_list)):
        dt = dt_list[j]
        if dt == 0.004:
            print("Original dt: 0.004 s, No downsampling needed")
            spec_list = ["2by2_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_4rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-02Nspline_k1.npz",
                         "3by3_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_6rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-4e-02Nspline_k1.npz",
                         "4by4_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_8rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-6e-02Nspline_k1.npz",
                         "6by6_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_12rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-01Nspline_k.npz",
                         "8by8_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_16rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-5e-01Nspline_k.npz",
                         "10by10_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_20rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-1e+00Nspline_.npz"]
        else:
            print(f"Downsampled dt: {dt} s")
            spec_list = [f"downsampled_{dt}_2by2.npz",
                        f"downsampled_{dt}_3by3.npz",
                        f"downsampled_{dt}_4by4.npz",
                        f"downsampled_{dt}_6by6.npz",
                        f"downsampled_{dt}_8by8.npz",
                        f"downsampled_{dt}_10by10.npz"]

        narma2_R2_train_scores_list = []
        narma2_R2_test_scores_list = []
        narma2_capacity_train_scores_list = []
        narma2_capacity_test_scores_list = []

        narma5_R2_train_scores_list = []
        narma5_R2_test_scores_list = []
        narma5_capacity_train_scores_list = []
        narma5_capacity_test_scores_list = []

        narma10_R2_train_scores_list = []
        narma10_R2_test_scores_list = []
        narma10_capacity_train_scores_list = []
        narma10_capacity_test_scores_list = []

        spec_label_list = []

        fignarma2, axnarma2 = plt.subplots(6, 2, figsize=(25, 25))
        fignarma5, axnarma5 = plt.subplots(6, 2, figsize=(25, 25))
        fignarma10, axnarma10 = plt.subplots(6, 2, figsize=(25, 25))

        for i in range(len(spec_list)):
            spec = spec_list[i]
            file_path = f'Simulations/SMASIS_sims/all_post_proc_1sf_YM100/'
            every_x_frame = 1
            sample_freq = np.rint(250/every_x_frame).astype(int)

            if dt == 0.004:
                spec_label = spec[:4] if spec[:4] != '10by' else spec[:6]
            else:
                spec_label = spec[-8:-8+4] if spec[-8:-8+4] != 'by10' else spec[-10:-8+4]
            spec_label_list.append(spec_label)
            print(spec_label)

            data = np.load(f"{file_path}{spec}", allow_pickle=True)

            input_data = data['input_data']
            
            # input_data = -1 + (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data)) * (1 - (-1))
            input_data = 0 + (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data)) * (0.2 - (0))
            output_data = data['output_data']
            time_data = data['time_data']

            regressor = "Rid"
            test_size = 0.25
            alpha = 0.0001

            # NARMA-2 testing
            narma2_RMSE_train, narma2_RMSE_test, narma2_R2_train, narma2_R2_test, y_train, y_train_pred, y_test, y_test_pred = narma2_test(input_data, output_data, regressor, test_size, alpha)

            narma2_R2_train_scores_list.append(narma2_R2_train)
            narma2_R2_test_scores_list.append(narma2_R2_test)
            narma2_capacity_train_scores_list.append(narma2_RMSE_train)
            narma2_capacity_test_scores_list.append(narma2_RMSE_test)

            print(f"NARMA-2 Training R2: {narma2_R2_train}, Testing R2: {narma2_R2_test}, RMSE Train: {narma2_RMSE_train}, RMSE Test: {narma2_RMSE_test}")

            axnarma2[i, 0].plot(y_train_pred, color='orange', label='Predicted')
            axnarma2[i, 0].plot(y_train, 'k--', label='True', alpha=0.7)
            axnarma2[i, 0].set_title(f'Training NARMA-2 {spec_label}')

            axnarma2[i, 1].plot(y_test_pred, color='orange', label='Predicted')
            axnarma2[i, 1].plot(y_test, 'k--', label='True', alpha=0.7)
            axnarma2[i, 1].set_title(f'Testing NARMA-2 {spec_label}')

            # NARMA-5 testing
            narma5_RMSE_train, narma5_RMSE_test, narma5_R2_train, narma5_R2_test, y_train, y_train_pred, y_test, y_test_pred = narma5_test(input_data, output_data, regressor, test_size, alpha)

            narma5_R2_train_scores_list.append(narma5_R2_train)
            narma5_R2_test_scores_list.append(narma5_R2_test)
            narma5_capacity_train_scores_list.append(narma5_RMSE_train)
            narma5_capacity_test_scores_list.append(narma5_RMSE_test)

            print(f"NARMA-5 Training R2: {narma5_R2_train}, Testing R2: {narma5_R2_test}, RMSE Train: {narma5_RMSE_train}, RMSE Test: {narma5_RMSE_test}")

            axnarma5[i, 0].plot(y_train_pred, color='orange', label='Predicted')
            axnarma5[i, 0].plot(y_train, 'k--', label='True', alpha=0.7)
            axnarma5[i, 0].set_title(f'Training NARMA-5 {spec_label}')

            axnarma5[i, 1].plot(y_test_pred, color='orange', label='Predicted')
            axnarma5[i, 1].plot(y_test, 'k--', label='True', alpha=0.7)
            axnarma5[i, 1].set_title(f'Testing NARMA-5 {spec_label}')
            
            # NARMA-10 testing
            narma10_RMSE_train, narma10_RMSE_test, narma10_R2_train, narma10_R2_test, y_train, y_train_pred, y_test, y_test_pred = narma10_test(input_data, output_data, regressor, test_size, alpha)

            narma10_R2_train_scores_list.append(narma10_R2_train)
            narma10_R2_test_scores_list.append(narma10_R2_test)
            narma10_capacity_train_scores_list.append(narma10_RMSE_train)
            narma10_capacity_test_scores_list.append(narma10_RMSE_test)

            print(f"NARMA-10 Training R2: {narma10_R2_train}, Testing R2: {narma10_R2_test}, RMSE Train: {narma10_RMSE_train}, RMSE Test: {narma10_RMSE_test}")

            axnarma10[i, 0].plot(y_train_pred, color='orange', label='Predicted')
            axnarma10[i, 0].plot(y_train, 'k--', label='True', alpha=0.7)
            axnarma10[i, 0].set_title(f'Training NARMA-10 {spec_label}')

            axnarma10[i, 1].plot(y_test_pred, color='orange', label='Predicted')
            axnarma10[i, 1].plot(y_test, 'k--', label='True', alpha=0.7)
            axnarma10[i, 1].set_title(f'Testing NARMA-10 {spec_label}')

            narma_sum = (narma2_R2_test + narma5_R2_test + narma10_R2_test)/3
            narma_score_sum_array[j, i] = narma_sum

        axnarma2[0, 1].legend()
        fignarma2.savefig(f'Simulations/SMASIS_sims/all_post_proc_1sf_YM100/narma2_dt{dt}.png')
        plt.close(fignarma2)
        
        axnarma5[0, 1].legend()
        fignarma5.savefig(f'Simulations/SMASIS_sims/all_post_proc_1sf_YM100/narma5_dt{dt}.png')
        plt.close(fignarma5)

        axnarma10[0, 1].legend()
        fignarma10.savefig(f'Simulations/SMASIS_sims/all_post_proc_1sf_YM100/narma10_dt{dt}.png')
        plt.close(fignarma10)

        plt.figure(figsize=(7.5, 7.5))
        plt.plot(spec_label_list, narma2_R2_test_scores_list, 'o-', label='NARMA-2 Test R2', markersize=10)
        plt.plot(spec_label_list, narma5_R2_test_scores_list, 'o-', label='NARMA-5 Test R2', markersize=10)
        plt.plot(spec_label_list, narma10_R2_test_scores_list, 'o-', label='NARMA-10 Test R2', markersize=10)
        plt.ylim(-0.1, 1.1)
        plt.title('NARMA Testing R2 Scores')
        plt.xlabel('Fiber Network Size')
        plt.ylabel('R2 Score')
        plt.legend()
        plt.savefig(f'Simulations/SMASIS_sims/all_post_proc_1sf_YM100/narma_all_testR2_dt{dt}.png')
        plt.close()

    plt.figure(figsize=(7.5, 7.5))
    ax = plt.axes()
    ax.matshow(narma_score_sum_array, cmap='RdYlBu_r')
    ax.set_xticks(ticks=np.arange(6))
    ax.set_xticklabels(['2by2', '3by3', '4by4', '6by6', '8by8', '10by10'])
    ax.set_yticks(ticks=np.arange(len(dt_list)))
    ax.set_yticklabels(dt_list)
    for (i, j), z in np.ndenumerate(narma_score_sum_array):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    cbar = plt.colorbar(ax.matshow(narma_score_sum_array, cmap='RdYlBu_r'))
    plt.xlabel('Fiber Network Size')
    plt.ylabel('Downsampled dt (s)')
    plt.title('NARMA Testing R2 Scores Heatmap')
    plt.savefig(f'Simulations/SMASIS_sims/all_post_proc_1sf_YM100/narma_all_testR2_heatmap.png')
    plt.close()





        