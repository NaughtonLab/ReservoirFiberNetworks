import pickle
import numpy as np
from scipy.special import legendre
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def nonlinearity_testing(input, output, leg_max_order, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    x = output
    capacity_train_list = []
    capacity_test_list = []
    R2_train_list = []
    R2_test_list = []
    for n in range(1, leg_max_order+1):
        leg = legendre(n)
        y = leg(input)
        shape_input = input.shape
        train_size = int(shape_input[0] * (1 - test_size))
        y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
        clf.fit(x_train, y_train)

        # Training
        y_train_pred = clf.predict(x_train)
        # print(y_train.shape, train_size, input.shape)

        MSE_train = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        capacity_train = 1 - MSE_train/y2
        R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

        # Testing
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 2)
        y_test_pred[idx] = np.mean(y)
        y_test[idx, 0] = np.mean(y)

        MSE_test = mean_squared_error(y_true=y_test, y_pred=y_test_pred) #1/(len(y_test)) * (y_test[:, 0]-y_test_pred)**2
        # print(y_test.shape, y_test_pred.shape, MSE_test.shape)
        # plt.plot((y_test[:, 0]-y_test_pred)**2)
        # plt.title(f'MSE Test {n}')
        # plt.show()
        capacity_test = 1 - MSE_test/y2
        R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

        if R2_test < 0:
            R2_test = 0
        if R2_train < 0:
            R2_train = 0
        if capacity_test < 0:
            capacity_test = 0
        if capacity_train < 0:
            capacity_train = 0

        # print("legendre", n, "MSE", MSE_train, MSE_test, "y2", y2, "capacity", capacity_train, capacity_test)

        # plt.figure(figsize=(15, 5))
        # plt.subplot(121)
        # plt.scatter(input[:train_size, :], y_train, label='True')
        # plt.scatter(input[:train_size, :],y_train_pred, label='Predicted')
        # plt.title(f'Training Legendre {n}')
        # plt.legend()
        # plt.grid()

        # plt.subplot(122)
        # plt.scatter(input[train_size:, :], y_test, label='True')
        # plt.scatter(input[train_size:, :], y_test_pred, label='Predicted')
        # plt.title(f'Testing Legendre {n}')
        # plt.legend()
        # plt.grid()

        # plt.show()     

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def memory_testing(input, output, max_timesteps_back, regressor, test_size, alpha, sample_freq):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

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

        y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
        clf.fit(x_train, y_train)

        # Training
        y_train_pred = clf.predict(x_train)

        MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        capacity_train = 1 - MSE/y2
        R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

        # Testing
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 1)
        y_test_pred[idx] = np.mean(y)
        y_test[idx, 0] = np.mean(y)
        y2_test = (1/len(y_test)) * np.sum((y_test-np.mean(y_test))**2)

        MSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
        capacity_test = 1 - MSE/y2_test
        R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

        if R2_test < 0:
            R2_test = 0
        if R2_train < 0:
            R2_train = 0
        if capacity_test < 0:
            capacity_test = 0
        if capacity_train < 0:
            capacity_train = 0

        # plt.figure(figsize=(15, 5))
        # plt.subplot(121)
        # plt.scatter(y_train, y_train, label='True')
        # plt.plot(y_train_pred, label='Predicted')
        # plt.title(f'Training Memory {n}')
        # plt.legend()
        # plt.grid()

        # plt.subplot(122)
        # plt.plot(y_test, label='True')
        # plt.plot(y_test_pred, label='Predicted')
        # plt.title(f'Testing Memory {n}')
        # plt.legend()
        # plt.grid()

        # plt.show()

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

        # print("memory", n, R2_train, R2_test, capacity_train, capacity_test)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def nonlinearity_memory_matrix(input, output, leg_max_order, max_timesteps_back, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    capacity_train_matrix = np.zeros((leg_max_order, max_timesteps_back+1))
    capacity_test_matrix = np.zeros((leg_max_order, max_timesteps_back+1))
    R2_train_matrix = np.zeros((leg_max_order, max_timesteps_back+1))
    R2_test_matrix = np.zeros((leg_max_order, max_timesteps_back+1))

    for n in range(1, leg_max_order+1):
        leg = legendre(n)
        for t in range(max_timesteps_back+1):
            x = output[t:]
            if t == 0:
                y = leg(input)
            else:
                y = leg(input[:-t])

            y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)
        
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
            clf.fit(x_train, y_train)

            # Training
            y_train_pred = clf.predict(x_train)

            MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
            capacity_train = 1 - MSE/y2
            R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

            # Testing
            y_test_pred = clf.predict(x_test)

            idx = np.where(abs(y_test_pred) > 2)
            y_test_pred[idx] = np.mean(y)
            y_test[idx, 0] = np.mean(y)

            MSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
            capacity_test = 1 - MSE/y2
            R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

            if R2_test < 0:
                R2_test = 0
            if R2_train < 0:
                R2_train = 0
            if capacity_test < 0:
                capacity_test = 0
            if capacity_train < 0:
                capacity_train = 0

            capacity_train_matrix[n-1, t] = capacity_train
            capacity_test_matrix[n-1, t] = capacity_test
            R2_train_matrix[n-1, t] = R2_train
            R2_test_matrix[n-1, t] = R2_test

    return capacity_train_matrix, capacity_test_matrix, R2_train_matrix, R2_test_matrix

def narma_test(input, output, n_list, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    else:
        print("Please specify the regressor")

    # normalizing input between 0 and 0.5
    x = output#[50:, :]
    input = (input - np.min(input)) / (np.max(input) - np.min(input)) * 0.5 + 0
    # input = input[50:, :]
    y = np.zeros((input.shape[0], input.shape[1]))
    
    capacity_train_list = []
    capacity_test_list = []
    R2_train_list = []
    R2_test_list = []
    for n in n_list:
        if n == 2:
            a = 0.4
            b = 0.4
            g = 0.6
            d = 0.1
            for t in range(2, input.shape[0]):
                alpha_term = a * y[t-1]
                beta_term = b * y[t-1] * y[t-2]
                gamma_term = g * input[t-1]**3
                y[t] = alpha_term + beta_term + gamma_term + d
            # for t in range(2, input.shape[0]):
            #     alpha_term = a * y[t-1]
            #     beta_term = b * y[t-1] * y[t-2]
            #     gamma_term = g * input[t]**3
            #     y[t] = alpha_term + beta_term + gamma_term + d
                # y[t] = np.clip(y[t], 0, 50000)  # Clip the values to be between 0 and 1

        else:
            a = 0.3
            b = 0.05
            g = 1.5
            d = 0.1
            for t in range(n, input.shape[0]):
                alpha_term = a * y[t-1]
                beta_term = b * y[t-1] * sum([y[t-i-1] for i in range(0, n)])
                gamma_term = g * input[t-1] * input[t-n]
                y[t] = alpha_term + beta_term + gamma_term + d

        y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
        clf.fit(x_train, y_train)

        # Training
        y_train_pred = clf.predict(x_train)

        MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        capacity_train = 1 - MSE/y2
        R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

        # Testing
        y_test_pred = clf.predict(x_test)

        MSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
        capacity_test = 1 - MSE/y2
        R2_test = r2_score(y_true=y_test, y_pred=y_test_pred)
        # if n == 2:
        #     plt.figure(figsize=(20, 5))
        #     plt.subplot(121)
        #     plt.plot(y_train, label='True')
        #     plt.plot(y_train_pred, label='Predicted')
        #     plt.title(f'Training NARMA {n}')
        #     plt.legend()
        #     plt.subplot(122)
        #     plt.plot(y_test, label='True')
        #     plt.plot(y_test_pred, label='Predicted')
        #     plt.title(f'Testing NARMA {n} Input')
        #     plt.legend()
        #     plt.show()

        #     print(f"Training NARMA {n} Capacity: {capacity_train}, R2: {R2_train}")
        #     print(f"Testing NARMA {n} Capacity: {capacity_test}, R2: {R2_test}")


        if R2_test < 0:
            R2_test = 0
        if R2_train < 0:
            R2_train = 0
        if capacity_test < 0:
            capacity_test = 0
        if capacity_train < 0:
            capacity_train = 0

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

if __name__ == '__main__':
    # spec = "2by2_post_proc_20samplingfreq_diffconstraints_mm_g_s_FiberSim_4rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-02Nspline_k1e+09_kt1e+09_"
    # spec = "3by3_post_proc_20samplingfreq_diffconstraints_mm_g_s_FiberSim_6rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-4e-02Nspline_k1e+09_kt1e+09_"
    # spec = "4by4_post_proc_20samplingfreq_diffconstraints_mm_g_s_FiberSim_8rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-6e-02Nspline_k1e+09_kt1e+09_"
    # spec = "6by6_post_proc_20samplingfreq_diffconstraints_mm_g_s_FiberSim_12rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-01Nspline_k1e+09_kt1e+09"
    
    # every 20th = 12.5 sf YM 10
    # spec_list = ["2by2_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_4rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-02Nspline_k.npz",
    #              "3by3_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_6rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-4e-02Nspli.npz",
    #              "4by4_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_8rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-6e-02Nspline_k.npz",
    #              "6by6_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_12rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-01Nspline_.npz",
    #              "8by8_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_16rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-5e-01Nspline_.npz",
    #              "10by10_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_20rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-1e+00Nsplin.npz"]
    # every 20th = 12.5 sf YM 100
    # spec_list = ["2by2_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_4rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-02Nspline_k.npz",
    #              "3by3_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_6rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-4e-02Nspline_.npz",
    #              "4by4_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_8rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-6e-02Nspline_.npz",
    #              "6by6_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_12rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-01Nspline.npz",
    #              "8by8_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_16rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-5e-01Nspline.npz",
    #              "10by10_post_proc_20sf_diffconstraints_mm_g_s_FiberSim_20rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-1e+00Nspli.npz"]
    
    # every 50th = 5 sf YM 10
    # spec_list = ["2by2_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_4rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-02Nspline_.npz",
    #              "3by3_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_6rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-4e-02Nspline_k1.npz",
    #              "4by4_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_8rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-6e-02Nspline_k.npz",
    #              "6by6_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_12rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-01Nspline.npz",
    #              "8by8_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_16rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-5e-01Nspline_.npz",
    #              "10by10_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_20rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-1e+00Nspli.npz"]

    # every 50th = 5 sf YM 100
    # spec_list = ["2by2_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_4rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-02Nspline_.npz",
    #              "3by3_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_6rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-4e-02Nspline_k.npz",
    #              "4by4_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_8rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-6e-02Nspline_k.npz",
    #              "6by6_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_12rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-01Nspline.npz",
    #              "8by8_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_16rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-5e-01Nspline_.npz",
    #              "10by10_post_proc_50sf_diffconstraints_mm_g_s_FiberSim_20rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-1e+00Nsplin.npz"]

    # every frame = 250 sf YM 10
    # spec_list = ["2by2_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_4rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-02Nspline.npz",
    #              "3by3_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_6rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-4e-02Nspline_k1e.npz",
    #              "4by4_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_8rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-6e-02Nspline_k1e.npz",
    #              "6by6_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_12rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-01Nspline_k1.npz",
    #              "8by8_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_16rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-5e-01Nspline_k1.npz",
    #              "10by10_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_20rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+07Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-1e+00Nspline_.npz"]

    # every frame = 250 sf YM 100
    # spec_list = ["2by2_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_4rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-02Nspline_k1.npz",
    #              "3by3_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_6rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-4e-02Nspline_k1.npz",
    #              "4by4_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_8rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-6e-02Nspline_k1.npz",
    #              "6by6_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_12rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-2e-01Nspline_k.npz",
    #              "8by8_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_16rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-5e-01Nspline_k.npz",
    #              "10by10_post_proc_1sf_diffconstraints_mm_g_s_FiberSim_20rods_15sec_L5.00e-01m_R1.00e-03m_dx10mm_YM1.00e+08Pa_Density1.00e+03kgmm-3_Damping10_TF1e-02N_PF-1e+00Nspline_.npz"]

    leg_R2_train_scores_list = []
    leg_R2_test_scores_list = []
    leg_capacity_train_scores_list = []
    leg_capacity_test_scores_list = []

    leg_R2_train_sum_list = []
    leg_R2_test_sum_list = []
    leg_capacity_train_sum_list = []
    leg_capacity_test_sum_list = []

    mem_R2_train_scores_list = []
    mem_R2_test_scores_list = []
    mem_capacity_train_scores_list = []
    mem_capacity_test_scores_list = []

    mem_R2_train_sum_list = []
    mem_R2_test_sum_list = []
    mem_capacity_train_sum_list = []
    mem_capacity_test_sum_list = []

    narma_R2_train_sum_list = []
    narma_R2_test_sum_list = []
    narma_capacity_train_sum_list = []
    narma_capacity_test_sum_list = []

    spec_label_list = []

    spec_list = ["test0",
                 "test1",
                 "test2",
                 "test3"]

    figleg, axleg = plt.subplots(2, 2, figsize=(20, 20))
    figmem, axmem = plt.subplots(2, 2, figsize=(20, 20))

    for spec in spec_list:
        file_path = f'./Simulations/Tests/'#f'SMASIS_sims/all_post_proc_1sf_YM10/'
        every_x_frame = 1
        sample_freq = np.rint(250/every_x_frame).astype(int)

        spec_label = spec #spec[:4] if spec[:4] != '10by' else spec[:6]
        spec_label_list.append(spec_label)
        print(spec_label)

        data = np.load(f"{file_path}{spec}.npz", allow_pickle=True)

        input_data = data['input_data']
        
        input_data = -1 + (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data)) * (1 - (-1))
        # plt.figure(figsize=(10, 5))
        # plt.plot(input_data, label= 'input')
        # # leg = legendre(5)
        # # # plt.scatter(input_data, leg(input_data), label='Legendre 5')
        # plt.legend()
        # plt.show()

        output_data = data['output_data']
        # plt.figure(figsize=(10, 5))
        # plt.plot(output_data, label= 'output')
        # plt.show()
        time_data = data['time_data']

        # Nonlinearity testing
        leg_max_order = 10
        regressor = "Rid"
        test_size = 0.25
        alpha = 1.0
        leg_capacity_train_list, leg_capacity_test_list, leg_R2_train_list, leg_R2_test_list = nonlinearity_testing(input_data, output_data, leg_max_order, regressor, test_size, alpha)
        
        # Memory testing
        max_time_back_seconds = 1
        max_timesteps_back = np.rint(sample_freq*max_time_back_seconds).astype(int)
        # max_timesteps_back = 10
        # max_time_back_seconds = np.rint(max_timesteps_back/sample_freq).astype(int)
        regressor = "Rid"
        test_size = 0.25
        alpha = 1.0
        mem_capacity_train_list, mem_capacity_test_list, mem_R2_train_list, mem_R2_test_list = memory_testing(input_data, output_data, max_timesteps_back, regressor, test_size, alpha, sample_freq)

        leg_R2_train_scores_list.append(leg_R2_train_list)
        leg_R2_test_scores_list.append(leg_R2_test_list)
        leg_capacity_train_scores_list.append(leg_capacity_train_list)
        leg_capacity_test_scores_list.append(leg_capacity_test_list)

        leg_R2_train_sum_list.append(sum(leg_R2_train_list)/len(leg_R2_train_list))
        leg_R2_test_sum_list.append(sum(leg_R2_test_list)/len(leg_R2_test_list))
        leg_capacity_train_sum_list.append(sum(leg_capacity_train_list)/len(leg_capacity_train_list))
        leg_capacity_test_sum_list.append(sum(leg_capacity_test_list)/len(leg_capacity_test_list))

        mem_R2_train_scores_list.append(mem_R2_train_list)
        mem_R2_test_scores_list.append(mem_R2_test_list)
        mem_capacity_train_scores_list.append(mem_capacity_train_list)
        mem_capacity_test_scores_list.append(mem_capacity_test_list)

        mem_R2_train_sum_list.append(sum(mem_R2_train_list)/len(mem_R2_train_list))
        mem_R2_test_sum_list.append(sum(mem_R2_test_list)/len(mem_R2_test_list))
        mem_capacity_train_sum_list.append(sum(mem_capacity_train_list)/len(mem_capacity_train_list))
        mem_capacity_test_sum_list.append(sum(mem_capacity_test_list)/len(mem_capacity_test_list))

        # Plotting Legendre and Memory
        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        plt.plot(np.linspace(1, leg_max_order, leg_max_order), leg_capacity_train_list, '-o', label='Training')
        plt.plot(np.linspace(1, leg_max_order, leg_max_order), leg_capacity_test_list, '-o', label='Testing')
        plt.xlabel('Legendre Polynomial Order')
        plt.ylabel('Capacity')
        plt.title('Nonlinearity Capacity')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid()
        
        plt.subplot(122)
        plt.plot(np.linspace(1, leg_max_order, leg_max_order), leg_R2_train_list, '-o', label='Training')
        plt.plot(np.linspace(1, leg_max_order, leg_max_order), leg_R2_test_list, '-o', label='Testing')
        plt.xlabel('Legendre Polynomial Order')
        plt.ylabel('R2 score')
        plt.title('Nonlinearity R2 score')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid()

        plt.savefig(f"{file_path}/{spec_label}_legendre_eval.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_capacity_train_list, '-o', label='Training')
        plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_capacity_test_list, '-o', label='Testing')
        plt.xlabel('Seconds in the Past')
        plt.ylabel('Capacity')
        plt.title('Memory Capacity')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid()

        plt.subplot(122)
        plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_R2_train_list, '-o', label='Training')
        plt.plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_R2_test_list, '-o', label='Testing')
        plt.xlabel('Seconds in the Past')
        plt.ylabel('R2 score')
        plt.title('Memory R2 score')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid()
        
        # plt.show()  # Show the plots
        plt.savefig(f"{file_path}/{spec_label}_memory_eval.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Nonlinearity and Memory testing
        # capacity_train_matrix, capacity_test_matrix, R2_train_matrix, R2_test_matrix = nonlinearity_memory_matrix(input_data, output_data, leg_max_order, max_timesteps_back, regressor, test_size, alpha)

        # Plotting the matrix
        # fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(20, 20))
        # ax1[0].matshow(capacity_train_matrix, cmap=matplotlib.colormaps['viridis'])
        # ax1[0].set_title("Training score")
        # ax1[0].set_ylabel("Legendre Polynomial Order")
        # ax1[0].xaxis.set_ticks_position('bottom')  # Move xticks to the bottom
        # ax1[0].xaxis.set_label_position('bottom')  # Move xlabel to the bottom
        # ax1[0].set_xlabel("Seconds in the Past")
        # ax1[0].set_yticks(np.arange(leg_max_order))
        # ax1[0].set_yticklabels(np.arange(1, leg_max_order + 1))
        # ax1[0].set_xticks(np.arange(max_timesteps_back+1))
        # ax1[0].set_xticklabels(np.round(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), 2))

        # ax1[1].matshow(capacity_test_matrix, cmap=matplotlib.colormaps['viridis'])
        # ax1[1].set_title("Testing score")
        # ax1[1].set_ylabel("Legendre Polynomial Order")
        # ax1[1].xaxis.set_ticks_position('bottom')  # Move xticks to the bottom
        # ax1[1].xaxis.set_label_position('bottom')  # Move xlabel to the bottom
        # ax1[1].set_xlabel("Seconds in the Past")
        # ax1[1].set_yticks(np.arange(leg_max_order))
        # ax1[1].set_yticklabels(np.arange(1, leg_max_order + 1))
        # ax1[1].set_xticks(np.arange(max_timesteps_back+1))
        # ax1[1].set_xticklabels(np.round(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), 2))

        # for i in range(1, leg_max_order+1):
        #     for j in range(max_timesteps_back+1):
        #         c_train = capacity_train_matrix[i-1, j]
        #         c_test = capacity_test_matrix[i-1, j]
        #         ax1[0].text(j, i-1, f"{c_train:.4f}", va='center', ha='center')
        #         ax1[1].text(j, i-1, f"{c_test:.4f}", va='center', ha='center')

        # ax2[0].matshow(R2_train_matrix, cmap=matplotlib.colormaps['viridis'])
        # ax2[0].set_title("Training score")
        # ax2[0].set_ylabel("Legendre Polynomial Order")
        # ax2[0].xaxis.set_ticks_position('bottom')  # Move xticks to the bottom
        # ax2[0].xaxis.set_label_position('bottom')  # Move xlabel to the bottom
        # ax2[0].set_xlabel("Seconds in the Past")
        # ax2[0].set_yticks(np.arange(leg_max_order))
        # ax2[0].set_yticklabels(np.arange(1, leg_max_order + 1))
        # ax2[0].set_xticks(np.arange(max_timesteps_back+1))
        # ax2[0].set_xticklabels(np.round(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), 2))

        # ax2[1].matshow(R2_test_matrix, cmap=matplotlib.colormaps['viridis'])
        # ax2[1].set_title("Testing score")
        # ax2[1].set_ylabel("Legendre Polynomial Order")
        # ax2[1].xaxis.set_ticks_position('bottom')  # Move xticks to the bottom
        # ax2[1].xaxis.set_label_position('bottom')  # Move xlabel to the bottom
        # ax2[1].set_xlabel("Seconds in the Past")
        # ax2[1].set_yticks(np.arange(leg_max_order))
        # ax2[1].set_yticklabels(np.arange(1, leg_max_order + 1))
        # ax2[1].set_xticks(np.arange(max_timesteps_back+1))
        # ax2[1].set_xticklabels(np.round(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), 2))

        # for i in range(1, leg_max_order+1):
        #     for j in range(max_timesteps_back+1):
        #         c_train = R2_train_matrix[i-1, j]
        #         c_test = R2_test_matrix[i-1, j]
        #         ax2[0].text(j, i-1, f"{c_train:.4f}", va='center', ha='center')
        #         ax2[1].text(j, i-1, f"{c_test:.4f}", va='center', ha='center')

        # plt.tight_layout()
        # plt.show()  # Show the plots
        # plt.savefig(f"{file_path}/{spec_label}_heatmap.png", dpi=300, bbox_inches='tight')
        # plt.close()

        # axleg[0, 0].plot(np.linspace(1, leg_max_order, leg_max_order), leg_capacity_train_list, '-o', label=spec_label)
        # axleg[0, 1].plot(np.linspace(1, leg_max_order, leg_max_order), leg_capacity_test_list, '-o', label=spec_label)
        # axleg[1, 0].plot(np.linspace(1, leg_max_order, leg_max_order), leg_R2_train_list, '-o', label=spec_label)
        # axleg[1, 1].plot(np.linspace(1, leg_max_order, leg_max_order), leg_R2_test_list, '-o', label=spec_label)

        # axmem[0, 0].plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_capacity_train_list, '-o', label=spec_label)
        # axmem[0, 1].plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_capacity_test_list, '-o', label=spec_label)
        # axmem[1, 0].plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_R2_train_list, '-o', label=spec_label)
        # axmem[1, 1].plot(np.linspace(0, max_time_back_seconds, max_timesteps_back+1), mem_R2_test_list, '-o', label=spec_label)

        # NARMA testing
        # narma_n_list = [2, 5, 10]
        # regressor = "Rid"
        # test_size = 0.25
        # alpha = 0.1
        # narma_capacity_train_list, narma_capacity_test_list, narma_R2_train_list, narma_R2_test_list = narma_test(input_data, output_data, narma_n_list, regressor, test_size, alpha)

        # Plotting NARMA
        # plt.figure(figsize=(15, 5))
        # plt.subplot(121)
        # plt.plot(narma_n_list, narma_capacity_train_list, '-o', label='Training')
        # plt.plot(narma_n_list, narma_capacity_test_list, '-o', label='Testing')
        # plt.xlabel('NARMA Order')
        # plt.ylabel('Capacity')
        # plt.title('NARMA Capacity')
        # plt.ylim(-0.1, 1.1)
        # plt.legend()
        # plt.grid()

        # plt.subplot(122)
        # plt.plot(narma_n_list, narma_R2_train_list, '-o', label='Training')
        # plt.plot(narma_n_list, narma_R2_test_list, '-o', label='Testing')
        # plt.xlabel('NARMA Order')
        # plt.ylabel('R2 score')')
        # plt.title('NARMA R2 score')
        # plt.ylim(-0.1, 1.1)
        # plt.legend()
        # plt.grid()

        # plt.show()  # Show the plots
    
    # axleg[0, 0].set_xlabel('Legendre Polynomial Order')
    # axleg[0, 0].set_ylabel('Capacity')
    # axleg[0, 0].set_title('Nonlinearity Capacity on Training Set')
    # axleg[0, 0].set_ylim(-0.1, 1.1)
    # axleg[0, 0].legend()
    # axleg[0, 0].grid()

    # axleg[0, 1].set_xlabel('Legendre Polynomial Order')
    # axleg[0, 1].set_ylabel('Capacity')
    # axleg[0, 1].set_title('Nonlinearity Capacity on Testing Set')
    # axleg[0, 1].set_ylim(-0.1, 1.1)
    # axleg[0, 1].legend()
    # axleg[0, 1].grid()

    # axleg[1, 0].set_xlabel('Legendre Polynomial Order')
    # axleg[1, 0].set_ylabel('R2 score')
    # axleg[1, 0].set_title('Nonlinearity R2 score on Training Set')
    # axleg[1, 0].set_ylim(-0.1, 1.1)
    # axleg[1, 0].legend()
    # axleg[1, 0].grid()

    # axleg[1, 1].set_xlabel('Legendre Polynomial Order')
    # axleg[1, 1].set_ylabel('R2 score')
    # axleg[1, 1].set_title('Nonlinearity R2 score on Testing Set')
    # axleg[1, 1].set_ylim(-0.1, 1.1)
    # axleg[1, 1].legend()
    # axleg[1, 1].grid()

    # figleg.savefig(f"{file_path}/legendre_eval_network_density_comparison.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # axmem[0, 0].set_xlabel('Seconds in the Past')
    # axmem[0, 0].set_ylabel('Capacity')
    # axmem[0, 0].set_title('Memory Capacity on Training Set')
    # axmem[0, 0].set_ylim(-0.1, 1.1)
    # axmem[0, 0].legend()
    # axmem[0, 0].grid()
    
    # axmem[0, 1].set_xlabel('Seconds in the Past')
    # axmem[0, 1].set_ylabel('Capacity')
    # axmem[0, 1].set_title('Memory Capacity on Testing Set')
    # axmem[0, 1].set_ylim(-0.1, 1.1)
    # axmem[0, 1].legend()
    # axmem[0, 1].grid()

    # axmem[1, 0].set_xlabel('Seconds in the Past')
    # axmem[1, 0].set_ylabel('R2 score')
    # axmem[1, 0].set_title('Memory R2 score on Training Set')
    # axmem[1, 0].set_ylim(-0.1, 1.1)
    # axmem[1, 0].legend()
    # axmem[1, 0].grid()

    # axmem[1, 1].set_xlabel('Seconds in the Past')
    # axmem[1, 1].set_ylabel('R2 score')
    # axmem[1, 1].set_title('Memory R2 score on Testing Set')
    # axmem[1, 1].set_ylim(-0.1, 1.1)
    # axmem[1, 1].legend()
    # axmem[1, 1].grid()
    
    # figmem.savefig(f"{file_path}/memory_eval_network_density_comparison.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # labels = spec_label_list
    labels = ['SMASIS', 'All Rotations', 'All Rotations No Tension', 'Shaker']

    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(labels, leg_capacity_train_sum_list, '-o', label='Training')
    plt.plot(labels, leg_capacity_test_sum_list, '-o', label='Testing')
    plt.xlabel('Network Size')
    plt.ylabel('Capacity')
    plt.title('Nonlinearity Capacity')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(labels, leg_R2_train_sum_list, '-o', label='Training')
    plt.plot(labels, leg_R2_test_sum_list, '-o', label='Testing')
    plt.xlabel('Network Size')
    plt.ylabel('R2 score')
    plt.title('Nonlinearity R2 score')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid()

    plt.savefig(f"{file_path}/legendre_sum_eval_test0to4.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(labels, mem_capacity_train_sum_list, '-o', label='Training')
    plt.plot(labels, mem_capacity_test_sum_list, '-o', label='Testing')
    plt.xlabel('Network Size')
    plt.ylabel('Capacity')
    plt.title('Memory Capacity')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(labels, mem_R2_train_sum_list, '-o', label='Training')
    plt.plot(labels, mem_R2_test_sum_list, '-o', label='Testing')
    plt.xlabel('Network Size')
    plt.ylabel('R2 score')
    plt.title('Memory R2 score')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid()

    plt.savefig(f"{file_path}/memory_sum_eval_test0to4.png", dpi=300, bbox_inches='tight')
    plt.close()