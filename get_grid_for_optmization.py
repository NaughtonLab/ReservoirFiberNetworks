import numpy as np
import pandas as pd

def get_input_threads_nodes(num_vertical_threads, num_horizontal_threads, thread_length, dx):

    n_elem = np.rint(thread_length/dx).astype(int)

    vert_connect_idx = np.linspace(0, n_elem+1, num_horizontal_threads+2)
    vert_connect_idx = vert_connect_idx[1:-1]
    for i in range(len(vert_connect_idx)):
        vert_connect_idx[i] = int(vert_connect_idx[i])
    
    hor_connect_idx = np.linspace(0, n_elem+1, num_vertical_threads+2)
    hor_connect_idx = hor_connect_idx[1:-1]
    for i in range(len(hor_connect_idx)):
        hor_connect_idx[i] = int(hor_connect_idx[i])

    if num_horizontal_threads == 0:
        print("No horizontal threads, no vibration possible")
        return None
    elif num_horizontal_threads == 1 and num_vertical_threads == 0:
        input_threads_nodes = [(0, n_elem//2)]
        return input_threads_nodes
    else:     
        input_idxs = (hor_connect_idx[1:] + hor_connect_idx[:-1])//2
        input_idxs = np.append(input_idxs, (hor_connect_idx[-1] + n_elem) // 2).astype(int)
        num_input_locations = np.rint(np.linspace(num_vertical_threads, 1, num_horizontal_threads)).astype(int)
        node_idx_list_by_threads = []
        for i in range(len(num_input_locations)):
            if i + num_input_locations[i] < len(input_idxs):
                start = i
            else:
                diff = (i + num_input_locations[i]) - len(input_idxs)
                start = i - diff
            node_idxs = (input_idxs[start:]).tolist()
            node_idx_list_by_threads.append(node_idxs)

        input_threads_nodes = []
        for i in range(len(node_idx_list_by_threads)):
            for j in range(len(node_idx_list_by_threads[i])):
                input_threads_nodes.append((i, node_idx_list_by_threads[i][j]))

        return input_threads_nodes
    
if __name__ == "__main__":
    '''Optimization Parameters from Simulation'''
    thread_length = 500e-3 # 1 m --> 1e3 mm
    thread_diameter = 2e-3 # 1 m --> 1e3 mm
    dx = 10e-3 # 1 m --> 1e3 mm

    num_horizontal_fibers_list = [5, 6, 7, 8, 9, 10]
    num_vertical_fibers_list = [5, 6, 7, 8, 9, 10]
    input_force = np.linspace(0.05, 1.0, 5)

    opt_params = pd.DataFrame(columns=['num_horizontal_threads', 'num_vertical_threads', 'input_thread', 'input_node', 'input_force'])

    for num_horizontal_threads in num_horizontal_fibers_list:
        for num_vertical_threads in num_vertical_fibers_list:
            input_threads_nodes = get_input_threads_nodes(num_vertical_threads, num_horizontal_threads, thread_length, dx)
            for i in range(len(input_threads_nodes)):
                thread = input_threads_nodes[i][0]
                node = input_threads_nodes[i][1]
                for force in input_force:
                    opt_params = pd.concat([opt_params, 
                                            pd.DataFrame({'num_horizontal_threads': [num_horizontal_threads], 
                                                          'num_vertical_threads': [num_vertical_threads], 
                                                          'input_thread': [thread], 
                                                          'input_node': [node], 
                                                          'input_force': [force]})], ignore_index=True)
                    
    print(len(opt_params))
    opt_params.to_csv('gridsearch_910_910.csv', index=False)

    #56 56 - 370
    #56 78 - 575
    #56 910 - 795
    #78 56 - 490
    #78 78 - 655
    #78 910 - 935
    #910 56 - 620
    #910 78 - 805
    #910 910 - 1020
                    