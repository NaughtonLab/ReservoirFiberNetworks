import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import legendre
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

matplotlib.rc('pdf', fonttype=42)

def nonlinearity_testing(input_data, output, leg_max_order, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    elif regressor == "Las":
        ### Lasso Regression
        clf = Lasso(alpha=alpha)
    elif regressor == "ElN":
        ### ElasticNet Regression
        clf = ElasticNet(alpha=alpha)
    else:
        print("Please specify the regressor")

    x = output
    capacity_train_list = []
    capacity_test_list = []
    R2_train_list = []
    R2_test_list = []
    for n in range(1, leg_max_order+1):
        leg = legendre(n)
        y = leg(input_data)
        shape_input = input_data.shape
        train_size = int(shape_input[0] * (1 - test_size))
        y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=True)
        
        # Training
        clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)

        idx = np.where(abs(y_train_pred) > 1)
        y_train_pred[idx] = np.mean(y)
        y_train[idx] = np.mean(y)

        # Testing
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 1)
        y_test_pred[idx] = np.mean(y)
        y_test[idx] = np.mean(y)

        y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)

        MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        capacity_train = 1 - MSE/y2_train
        R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

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

        # Wout_list.append(clf.coef_)  

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

    # np.savez(f"./Wout_list_nonlinearity_{regressor}.npz", Wout_list=Wout_list)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def memory_testing(input, output, max_timesteps_back, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    elif regressor == "Las":
        ### Lasso Regression
        clf = Lasso(alpha=alpha)
    elif regressor == "ElN":
        ### ElasticNet Regression
        clf = ElasticNet(alpha=alpha)
    else:
        print("Please specify the regressor")

    capacity_train_list = []
    capacity_test_list = []
    R2_train_list = []
    R2_test_list = []
    Wout_list = []
    for n in range(0, max_timesteps_back+1):
        x = output[n:]
        if n == 0:
            y = input
        else:
            y = input[:-n]

        y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
        
        # Training
        clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)

        idx = np.where(abs(y_train_pred) > 1)
        y_train_pred[idx] = np.mean(y)
        y_train[idx] = np.mean(y)

        # Testing
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 1)
        y_test_pred[idx] = np.mean(y)
        y_test[idx] = np.mean(y)

        y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)

        MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        capacity_train = 1 - MSE/y2_train
        R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

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

        # Wout_list.append(clf.coef_) 

        capacity_train_list.append(capacity_train)
        capacity_test_list.append(capacity_test)
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)

        # print("memory", n, R2_train, R2_test, capacity_train, capacity_test)
    
    # np.savez(f"./Wout_list_memory_{regressor}.npz", Wout_list=Wout_list)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def memory_testing_less(input, output, max_timesteps_back, regressor, test_size, alpha):
    if regressor == "Lin":
        ### Linear Regression
        clf = LinearRegression()
    elif regressor == "Rid":
        ### Ridge Regression
        clf = Ridge(alpha=alpha)
    elif regressor == "Las":
        ### Lasso Regression
        clf = Lasso(alpha=alpha)
    elif regressor == "ElN":
        ### ElasticNet Regression
        clf = ElasticNet(alpha=alpha)
    else:
        print("Please specify the regressor")

    capacity_train_list = []
    capacity_test_list = []
    R2_train_list = []
    R2_test_list = []
    Wout_list = []
    for n in max_timesteps_back:
        x = output[n:]
        if n == 0:
            y = input
        else:
            y = input[:-n]

        y2 = (1/len(y)) * np.sum((y-np.mean(y))**2)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
        
        # Training
        clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)

        idx = np.where(abs(y_train_pred) > 1)
        y_train_pred[idx] = np.mean(y)
        y_train[idx] = np.mean(y)

        # Testing
        y_test_pred = clf.predict(x_test)

        idx = np.where(abs(y_test_pred) > 1)
        y_test_pred[idx] = np.mean(y)
        y_test[idx] = np.mean(y)

        y2_train = (1/len(y_train)) * np.sum((y_train-np.mean(y_train))**2)

        MSE = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        capacity_train = 1 - MSE/y2_train
        R2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

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
    
    # if regressor == "Rid":
    #     np.savez("./Wout_list_memory.npz", Wout_list=Wout_list)

    return capacity_train_list, capacity_test_list, R2_train_list, R2_test_list

def groups_of_six_output_selection(conns, hors, vers, ip, leg_max_order, max_timesteps_back, regressor, test_size, alpha, leg_x, mem_x):
    conns_x = conns[:, ::2]
    conns_y = conns[:, 1::2]
    hors_x = hors[:, ::2]
    hors_y = hors[:, 1::2]
    vers_x = vers[:, ::2]
    vers_y = vers[:, 1::2]

    groups_dict = {
        'Conns X': [conns_x],
        'Conns Y': [conns_y],
        'Hors X': [hors_x],
        'Hors Y': [hors_y],
        'Vers X': [vers_x],
        'Vers Y': [vers_y]}
    
    fig_leg, ax_leg = plt.subplots(1, 1, figsize=(6, 6))
    fig_mem, ax_mem = plt.subplots(1, 1, figsize=(6, 6))
    
    for group_name, group_outputs in groups_dict.items():
        leg_capacity_train, leg_capacity_test, leg_R2_train, leg_R2_test = nonlinearity_testing(
            ip, group_outputs[0], leg_max_order, regressor, test_size, alpha)
        mem_capacity_train, mem_capacity_test, mem_R2_train, mem_R2_test = memory_testing(
            ip, group_outputs[0], max_timesteps_back, regressor, test_size, alpha)
        
        groups_dict[group_name].append(leg_capacity_train)
        groups_dict[group_name].append(leg_capacity_test)
        groups_dict[group_name].append(mem_capacity_train)
        groups_dict[group_name].append(mem_capacity_test)

        ax_leg.plot(leg_x, leg_capacity_test, '-o', label=group_name)
        ax_mem.plot(mem_x, mem_capacity_test, '-o', label=group_name)

    ax_leg.set_xlabel('Legendre Polynomial Order')       
    ax_mem.set_xlabel('Time (s)')
    ax_leg.set_ylabel('Capacity')
    ax_mem.set_ylabel('Capacity')
    ax_leg.legend()
    ax_mem.legend()

    return groups_dict, fig_leg, fig_mem, ax_leg, ax_mem

def combinations_of_two_groups_output_selection(groups_dict, ip, leg_max_order, max_timesteps_back, regressor, test_size, alpha):
   
    group_names = list(groups_dict.keys())
    num_groups = len(group_names)

    onc_train_list = []
    omc_train_list = []
    onc_test_list = []
    omc_test_list = []
    combination_list = []

    for i in range(num_groups):
        for j in range(i+1, num_groups):
            group_name_1 = group_names[i]
            group_name_2 = group_names[j]
            combined_outputs = np.hstack((groups_dict[group_name_1][0], groups_dict[group_name_2][0]))

            leg_capacity_train, leg_capacity_test, leg_R2_train, leg_R2_test = nonlinearity_testing(
                ip, combined_outputs, leg_max_order, regressor, test_size, alpha)
            mem_capacity_train, mem_capacity_test, mem_R2_train, mem_R2_test = memory_testing(
                ip, combined_outputs, max_timesteps_back, regressor, test_size, alpha)
            
            onc_train_list.append(sum(leg_capacity_train)/len(leg_capacity_train))
            omc_train_list.append(sum(mem_capacity_train)/len(mem_capacity_train))
            onc_test_list.append(sum(leg_capacity_test)/len(leg_capacity_test))
            omc_test_list.append(sum(mem_capacity_test)/len(mem_capacity_test))

            combined_group_name = f"{group_name_1} + {group_name_2}"
            combination_list.append(combined_group_name)

    max_onc_train = max(onc_train_list)
    max_onc_train_idx = onc_train_list.index(max_onc_train)
    combination_max_onc = combination_list[max_onc_train_idx]

    max_onc_test = max(onc_test_list)
    max_onc_test_idx = onc_test_list.index(max_onc_test)
    combination_max_onc_test = combination_list[max_onc_test_idx]
    
    max_omc_train = max(omc_train_list)
    max_omc_train_idx = omc_train_list.index(max_omc_train)
    combination_max_omc = combination_list[max_omc_train_idx]

    max_omc_test = max(omc_test_list)
    max_omc_test_idx = omc_test_list.index(max_omc_test)
    combination_max_omc_test = combination_list[max_omc_test_idx]

    print(f"Combination with highest Nonlinear Capacity (train): {combination_max_onc} with ONC = {max_onc_train}")
    print(f"Combination with highest Nonlinear Capacity (test): {combination_max_onc_test} with ONC = {max_onc_test}")
    print(f"Combination with highest Memory Capacity (train): {combination_max_omc} with OMC = {max_omc_train}")
    print(f"Combination with highest Memory Capacity (test): {combination_max_omc_test} with OMC = {max_omc_test}")

    best_groups = {'Max_ONC_train': [combination_max_onc, leg_capacity_train, leg_capacity_test, mem_capacity_train, mem_capacity_test],
                   'Max_OMC_train': [combination_max_omc, leg_capacity_train, leg_capacity_test, mem_capacity_train, mem_capacity_test],
                   'Max_ONC_test': [combination_max_onc_test, leg_capacity_train, leg_capacity_test, mem_capacity_train, mem_capacity_test],
                   'Max_OMC_test': [combination_max_omc_test, leg_capacity_train, leg_capacity_test, mem_capacity_train, mem_capacity_test]}
    
    return best_groups 

def near_springs_output_selection(conns, hors, vers, num_horizontal_threads, num_vertical_threads):
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
        if ver_idx == num_vertical_threads:
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
        elif hor_idx == 4 and ver_idx >= 2:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])
        elif hor_idx == 5 and ver_idx >= 1:
            near_springs_outputs.append(hors[:, i])
            near_springs_outputs.append(hors[:, i+1])

        ver_idx += 1
        if ver_idx == num_vertical_threads + 1:
            ver_idx = 0
            hor_idx += 1

    hor_idx = 0
    ver_idx = 0

    for i in range(0, vers.shape[1], 2):
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
        if hor_idx == num_horizontal_threads + 1:
            hor_idx = 0
            ver_idx += 1

    near_springs_outputs_arr = np.hstack([arr[:, np.newaxis] for arr in near_springs_outputs])

    return near_springs_outputs_arr

def near_actuation_output_selection(conns, hors, vers, num_horizontal_threads, num_vertical_threads):
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
        if ver_idx == num_vertical_threads:
            ver_idx = 0
            hor_idx += 1

    hor_idx = 0
    ver_idx = 0

    for i in range(0, hors.shape[1], 2):
        if (hor_idx >= 1 and hor_idx <= 4) and (ver_idx >= 2 and ver_idx <= 4):
            near_actuation_outputs.append(hors[:, i])
            near_actuation_outputs.append(hors[:, i+1])

        ver_idx += 1
        if ver_idx == num_vertical_threads + 1:
            ver_idx = 0
            hor_idx += 1

    hor_idx = 0
    ver_idx = 0

    for i in range(0, vers.shape[1], 2):
        if (ver_idx >= 1 and ver_idx <= 4) and (hor_idx >= 2 and hor_idx <= 4):
            near_actuation_outputs.append(vers[:, i])
            near_actuation_outputs.append(vers[:, i+1])
        
        hor_idx += 1
        if hor_idx == num_horizontal_threads + 1:
            hor_idx = 0
            ver_idx += 1

    near_actuation_outputs_arr = np.hstack([arr[:, np.newaxis] for arr in near_actuation_outputs])

    return near_actuation_outputs_arr

if __name__ == "__main__":
    num_horizontal_threads = 6
    num_vertical_threads = num_horizontal_threads

    current_path = os.path.dirname(__file__)
    save_path = "C:/D/ABHYAAS/OneDrive - Virginia Tech/naughtonlab - active_projects/2025-JIMSS_paper/pdf_v4/Fig 3"

    orig = np.load(f"{current_path}/{num_horizontal_threads}by{num_vertical_threads}_orig.npz", allow_pickle=True)

    leg = 'legendre_cap'
    mem = 'memory_cap'

    regressor = "Rid"
    test_size = 0.25
    alpha = 1e-1

    leg_max_order = 10
    max_time_back_seconds = 1
    max_timesteps_back = np.rint(250*max_time_back_seconds).astype(int)
    leg_x = np.linspace(1, leg_max_order, leg_max_order)
    mem_x = np.linspace(0, max_time_back_seconds, max_timesteps_back+1)

    data = orig
    ip = data['input_data']
    ip = -1 + (ip - np.min(ip)) / (np.max(ip) - np.min(ip)) * (1 - (-1))

    op = data['output_data']
    op /= np.mean(op, axis=0)
    op /= np.std(op, axis=0)
    scaler = preprocessing.StandardScaler().fit(op)
    op = scaler.transform(op)

    leg_capacity_train_orig, leg_capacity_test_orig, leg_R2_train_orig, leg_R2_test_orig = nonlinearity_testing(
        ip, op, leg_max_order, regressor, test_size, alpha)
    
    mem_capacity_train_orig, mem_capacity_test_orig, mem_R2_train_orig, mem_R2_test_orig = memory_testing(
        ip, op, max_timesteps_back, regressor, test_size, alpha)
    
    onc_train_orig = sum(leg_capacity_train_orig)/len(leg_capacity_train_orig)
    omc_train_orig = sum(mem_capacity_train_orig)/len(mem_capacity_train_orig)
    
    onc_test_orig = sum(leg_capacity_test_orig)/len(leg_capacity_test_orig)
    omc_test_orig = sum(mem_capacity_test_orig)/len(mem_capacity_test_orig)

    print(f"Original Nonlinear Capacity (train): {onc_train_orig}, (test): {onc_test_orig}")
    print(f"Original Memory Capacity (train): {omc_train_orig}, (test): {omc_test_orig}")

    conns_end = num_horizontal_threads*num_vertical_threads*2
    hors_end = conns_end + ((num_vertical_threads+1)*num_horizontal_threads)*2
    vers_end = op.shape[1]

    conns = op[:, 0:conns_end]
    hors = op[:, conns_end:hors_end]
    vers = op[:, hors_end:vers_end]

    # groups_dict, fig_leg, fig_mem, ax_leg, ax_mem = groups_of_six_output_selection(
    #     conns, hors, vers, ip, leg_max_order, max_timesteps_back, regressor, test_size, alpha, leg_x, mem_x)
    
    # ax_leg.plot(leg_x, leg_capacity_test_orig, '-o', label='All Outputs', color='black', linewidth=3, zorder=0)
    # ax_mem.plot(mem_x, mem_capacity_test_orig, '-o', label='All Outputs', color='black', linewidth=3, zorder=0)
    # ax_leg.set_title('Nonlinear Capacity - Groups of Six Outputs')
    # ax_mem.set_title('Memory Capacity - Groups of Six Outputs')
    # ax_leg.legend()
    # ax_mem.legend()

    # fig_leg.savefig(f"{save_path}/Groups_of_Six_Nonlinear_Capacity_{regressor}.pdf", dpi=300)
    # fig_mem.savefig(f"{save_path}/Groups_of_Six_Memory_Capacity_{regressor}.pdf", dpi=300)

    # plt.close('all')

    # onc_train_list = []
    # omc_train_list = []
    # onc_test_list = []
    # omc_test_list = []

    # for group_name, group_data in groups_dict.items():
    #     leg_capacity_train = group_data[1]
    #     leg_capacity_test = group_data[2]
    #     mem_capacity_train = group_data[3]
    #     mem_capacity_test = group_data[4]

    #     onc_train_list.append(sum(leg_capacity_train)/len(leg_capacity_train))
    #     omc_train_list.append(sum(mem_capacity_train)/len(mem_capacity_train))
    #     onc_test_list.append(sum(leg_capacity_test)/len(leg_capacity_test))
    #     omc_test_list.append(sum(mem_capacity_test)/len(mem_capacity_test))

    # max_onc_train = max(onc_train_list)
    # max_onc_train_idx = onc_train_list.index(max_onc_train)
    # group_max_onc = list(groups_dict.keys())[max_onc_train_idx]
    # max_onc_test = max(onc_test_list)
    # max_onc_test_idx = onc_test_list.index(max_onc_test)
    # group_max_onc_test = list(groups_dict.keys())[max_onc_test_idx]
    # max_omc_train = max(omc_train_list)
    # max_omc_train_idx = omc_train_list.index(max_omc_train)
    # group_max_omc = list(groups_dict.keys())[max_omc_train_idx]
    # max_omc_test = max(omc_test_list)
    # max_omc_test_idx = omc_test_list.index(max_omc_test)
    # group_max_omc_test = list(groups_dict.keys())[max_omc_test_idx]

    # print("Results for Groups of Six Output Selection:")
    # print("Nonlinear capacities of all groups:", onc_test_list)
    # print("Memory capacities of all groups:", omc_test_list)

    # print(f"Group with highest Nonlinear Capacity (train): {group_max_onc} with ONC = {max_onc_train}")
    # print(f"Group with highest Nonlinear Capacity (test): {group_max_onc_test} with ONC = {max_onc_test}")
    # print(f"Group with highest Memory Capacity (train): {group_max_omc} with OMC = {max_omc_train}")
    # print(f"Group with highest Memory Capacity (test): {group_max_omc_test} with OMC = {max_omc_test}")

    # fig_leg_bar1, ax_leg_bar1 = plt.subplots(1, 1, figsize=(7.5, 7.5))
    # fig_mem_bar1, ax_mem_bar1 = plt.subplots(1, 1, figsize=(7.5, 7.5))
    # categories = ['All outputs'] + list(groups_dict.keys())
    # x = np.arange(len(categories))
    # width = 0.35
    # onc_plot_values = [onc_test_orig] + onc_test_list
    # onc_plot_values = [onc_plot_values[i]/onc_test_orig for i in range(len(onc_plot_values))]
    # omc_plot_values = [omc_test_orig] + omc_test_list
    # omc_plot_values = [omc_plot_values[i]/omc_test_orig for i in range(len(omc_plot_values))]

    # ax_leg_bar1.bar(x, onc_plot_values, width)
    # ax_mem_bar1.bar(x, omc_plot_values, width)
    # ax_leg_bar1.set_ylabel('Normalized Nonlinear Capacity')
    # ax_mem_bar1.set_ylabel('Normalized Memory Capacity')
    # ax_leg_bar1.set_xticks(x)
    # ax_mem_bar1.set_xticks(x)
    # ax_leg_bar1.set_xticklabels(categories, rotation=15)
    # ax_mem_bar1.set_xticklabels(categories, rotation=15)
    # ax_leg_bar1.set_title("Comparison of Nonlinearity Performance")
    # ax_mem_bar1.set_title("Comparison of Memory Performance")

    # # fig_leg_bar1.savefig(f"{save_path}/Nonlinearity_Comparison_of_6_groups.pdf", dpi=300)
    # # fig_mem_bar1.savefig(f"{save_path}/Memory_Comparison_of_6_groups.pdf", dpi=300)

    # plt.close('all')

    # best_groups = combinations_of_two_groups_output_selection(
    #     groups_dict, ip, leg_max_order, max_timesteps_back, regressor, test_size, alpha)

    vers_x = vers[:, ::2]
    hors_y = hors[:, 1::2]

    leg_capacity_train_hy, leg_capacity_test_hy, leg_R2_train_hy, leg_R2_test_hy = nonlinearity_testing(
        ip, hors_y, leg_max_order, regressor, test_size, alpha)
    mem_capacity_train_hy, mem_capacity_test_hy, mem_R2_train_hy, mem_R2_test_hy = memory_testing(
        ip, hors_y, max_timesteps_back, regressor, test_size, alpha)
    
    onc_train_hy = sum(leg_capacity_train_hy)/len(leg_capacity_train_hy)
    omc_train_hy = sum(mem_capacity_train_hy)/len(mem_capacity_train_hy)
    onc_test_hy = sum(leg_capacity_test_hy)/len(leg_capacity_test_hy)
    omc_test_hy = sum(mem_capacity_test_hy)/len(mem_capacity_test_hy)

    leg_capacity_train_vx, leg_capacity_test_vx, leg_R2_train_vx, leg_R2_test_vx = nonlinearity_testing(
        ip, vers_x, leg_max_order, regressor, test_size, alpha)
    mem_capacity_train_vx, mem_capacity_test_vx, mem_R2_train_vx, mem_R2_test_vx = memory_testing(
        ip, vers_x, max_timesteps_back, regressor, test_size, alpha)
    
    onc_train_vx = sum(leg_capacity_train_vx)/len(leg_capacity_train_vx)
    omc_train_vx = sum(mem_capacity_train_vx)/len(mem_capacity_train_vx)
    onc_test_vx = sum(leg_capacity_test_vx)/len(leg_capacity_test_vx)
    omc_test_vx = sum(mem_capacity_test_vx)/len(mem_capacity_test_vx)

    hors_y_vers_x = np.hstack((hors_y, vers_x))

    leg_capacity_train_hyvx, leg_capacity_test_hyvx, leg_R2_train_hyvx, leg_R2_test_hyvx = nonlinearity_testing(
        ip, hors_y_vers_x, leg_max_order, regressor, test_size, alpha)
    mem_capacity_train_hyvx, mem_capacity_test_hyvx, mem_R2_train_hyvx, mem_R2_test_hyvx = memory_testing(
        ip, hors_y_vers_x, max_timesteps_back, regressor, test_size, alpha)
    
    onc_train_hyvx = sum(leg_capacity_train_hyvx)/len(leg_capacity_train_hyvx)
    omc_train_hyvx = sum(mem_capacity_train_hyvx)/len(mem_capacity_train_hyvx)
    onc_test_hyvx = sum(leg_capacity_test_hyvx)/len(leg_capacity_test_hyvx)
    omc_test_hyvx = sum(mem_capacity_test_hyvx)/len(mem_capacity_test_hyvx)

    print(f"Hors Y + Vers X Nonlinear Capacity (train): {onc_train_hyvx}, (test): {onc_test_hyvx}")
    print(f"Hors Y + Vers X Memory Capacity (train): {omc_train_hyvx}, (test): {omc_test_hyvx}")

    near_springs_outputs = near_springs_output_selection(
        conns, hors, vers, num_horizontal_threads, num_vertical_threads)
    # print(near_springs_outputs.shape) # 67 * 2
    
    leg_capacity_train_ns, leg_capacity_test_ns, leg_R2_train_ns, leg_R2_test_ns = nonlinearity_testing(
        ip, near_springs_outputs, leg_max_order, regressor, test_size, alpha)
    mem_capacity_train_ns, mem_capacity_test_ns, mem_R2_train_ns, mem_R2_test_ns = memory_testing(
        ip, near_springs_outputs, max_timesteps_back, regressor, test_size, alpha)
    
    onc_train_ns = sum(leg_capacity_train_ns)/len(leg_capacity_train_ns)
    omc_train_ns = sum(mem_capacity_train_ns)/len(mem_capacity_train_ns)
    onc_test_ns = sum(leg_capacity_test_ns)/len(leg_capacity_test_ns)
    omc_test_ns = sum(mem_capacity_test_ns)/len(mem_capacity_test_ns)

    print(f"Near Springs Nonlinear Capacity (train): {onc_train_ns}, (test): {onc_test_ns}")
    print(f"Near Springs Memory Capacity (train): {omc_train_ns}, (test): {omc_test_ns}")

    near_actuation_outputs = near_actuation_output_selection(
        conns, hors, vers, num_horizontal_threads, num_vertical_threads)
    # print(near_actuation_outputs.shape) # 40 * 2
    
    leg_capacity_train_na, leg_capacity_test_na, leg_R2_train_na, leg_R2_test_na = nonlinearity_testing(
        ip, near_actuation_outputs, leg_max_order, regressor, test_size, alpha)
    mem_capacity_train_na, mem_capacity_test_na, mem_R2_train_na, mem_R2_test_na = memory_testing(
        ip, near_actuation_outputs, max_timesteps_back, regressor, test_size, alpha)
    
    onc_train_na = sum(leg_capacity_train_na)/len(leg_capacity_train_na)
    omc_train_na = sum(mem_capacity_train_na)/len(mem_capacity_train_na)
    onc_test_na = sum(leg_capacity_test_na)/len(leg_capacity_test_na)
    omc_test_na = sum(mem_capacity_test_na)/len(mem_capacity_test_na)

    print(f"Near Actuation Nonlinear Capacity (train): {onc_train_na}, (test): {onc_test_na}")
    print(f"Near Actuation Memory Capacity (train): {omc_train_na}, (test): {omc_test_na}")

    fig_leg_all, ax_leg_all = plt.subplots(1, 1, figsize=(6, 6))
    fig_mem_all, ax_mem_all = plt.subplots(1, 1, figsize=(6, 6))
    fig_leg_bar, ax_leg_bar = plt.subplots(1, 1, figsize=(8, 6))
    fig_mem_bar, ax_mem_bar = plt.subplots(1, 1, figsize=(8, 6))

    labels = ['All Outputs', 'Hors Y + Vers X', 'Near Springs', 'Near Actuation']
    leg_capacity_tests = [leg_capacity_test_orig, leg_capacity_test_hyvx, 
                            leg_capacity_test_ns, leg_capacity_test_na]
    mem_capacity_tests = [mem_capacity_test_orig, mem_capacity_test_hyvx, 
                            mem_capacity_test_ns, mem_capacity_test_na]
    
    for i in range(len(labels)):
        ax_leg_all.plot(leg_x, leg_capacity_tests[i], '-o', label=labels[i])
        ax_mem_all.plot(mem_x, mem_capacity_tests[i], '-o', label=labels[i])

    ax_leg_all.set_xlabel('Legendre Polynomial Order')       
    ax_mem_all.set_xlabel('Time (s)')
    ax_leg_all.set_ylabel('Capacity')
    ax_mem_all.set_ylabel('Capacity')
    ax_leg_all.set_title('Nonlinear Capacity - Feature Selection')
    ax_mem_all.set_title('Memory Capacity - Feature Selection')
    ax_leg_all.legend()
    ax_mem_all.legend()
    fig_leg_all.savefig(f"{save_path}/Feature_Selection_Nonlinear_Capacity.pdf", dpi=300)
    fig_mem_all.savefig(f"{save_path}/Feature_Selection_Memory_Capacity.pdf", dpi=300)
    
    x = np.arange(len(labels))
    width = 0.35
    onc_tests = [onc_test_orig, onc_test_hyvx, onc_test_ns, onc_test_na]
    onc_tests = [onc_tests[i]/onc_test_orig for i in range(len(onc_tests))]
    omc_tests = [omc_test_orig, omc_test_hyvx, omc_test_ns, omc_test_na]
    omc_tests = [omc_tests[i]/omc_test_orig for i in range(len(omc_tests))]
    ax_leg_bar.bar(x, onc_tests, width, label='ONC')
    ax_mem_bar.bar(x, omc_tests, width, label='OMC')
    ax_leg_bar.set_xlabel('Feature Selection Method')
    ax_mem_bar.set_xlabel('Feature Selection Method')
    ax_leg_bar.set_ylabel('Overall Nonlinear Capacity (ONC)')
    ax_mem_bar.set_ylabel('Overall Memory Capacity (OMC)')
    ax_leg_bar.set_title('Overall Nonlinear Capacity (ONC) - Feature Selection')
    ax_mem_bar.set_title('Overall Memory Capacity (OMC) - Feature Selection')
    ax_leg_bar.set_xticks(x)
    ax_leg_bar.set_xticklabels(labels, rotation=15)
    ax_mem_bar.set_xticks(x)
    ax_mem_bar.set_xticklabels(labels, rotation=15)
    fig_leg_bar.savefig(f"{save_path}/Feature_Selection_Overall_Nonlinear_Capacity.pdf", dpi=300)
    fig_mem_bar.savefig(f"{save_path}/Feature_Selection_Overall_Memory_Capacity.pdf", dpi=300)

    plt.close('all')
