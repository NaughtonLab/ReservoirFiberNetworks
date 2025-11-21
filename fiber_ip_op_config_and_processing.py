import os
import pickle
import multiprocessing
import json
import numpy as np

from fiber_simulation_test0 import fiber_simulation
from utils.unit_scaling import unit_scaling

def wrapper_launcher(params):
    try:
        sim = fiber_simulation(**params)
        sim.launch_sim()
    except Exception as e:
        print(f'Something failed!, Error: {e}')
    return

def mp_handler(params_list):
    p = multiprocessing.Pool(1)
    p.map(wrapper_launcher, params_list)

def get_input_node_positions(num_vertical_threads, num_horizontal_threads, thread_length, dx, network_origin=np.zeros((2,))):

    n_elem = np.rint(thread_length/dx).astype(int)

    vert_connect_idx = np.linspace(0, n_elem+1, num_horizontal_threads+2)
    vert_connect_idx = vert_connect_idx[1:-1]
    for i in range(len(vert_connect_idx)):
        vert_connect_idx[i] = int(vert_connect_idx[i])
    
    hor_connect_idx = np.linspace(0, n_elem+1, num_vertical_threads+2)
    hor_connect_idx = hor_connect_idx[1:-1]
    for i in range(len(hor_connect_idx)):
        hor_connect_idx[i] = int(hor_connect_idx[i])

    horizontal_thread_y_pos = vert_connect_idx * dx - thread_length/2 + network_origin[1]
    input_y_positions = horizontal_thread_y_pos[::-1]

    if num_horizontal_threads == 0:
        print("No horizontal threads, no vibration possible")
        return None
    elif num_horizontal_threads == 1 and num_vertical_threads == 0:
        input_positions = np.array([[n_elem*dx/2 - thread_length/2 + network_origin[0], 0.0]])
        return input_positions
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

        input_x_positions = -1234567890*np.ones((num_horizontal_threads, num_vertical_threads))
        for j in range(input_x_positions.shape[0]):
            node_idxs = node_idx_list_by_threads[j]
            if len(input_idxs) == len(node_idxs):
                for i in range(input_x_positions.shape[1]):
                    x_pos = node_idxs[i]*dx - thread_length/2 + network_origin[0]
                    input_x_positions[j, i] = x_pos
            elif len(input_idxs) > len(node_idxs):
                diff = abs(len(input_idxs) - len(node_idxs))
                for i in range(diff, input_x_positions.shape[1]):
                    x_pos = node_idxs[i-diff]*dx - thread_length/2 + network_origin[0]
                    input_x_positions[j, i] = x_pos
            else:
                continue

        num_input_pos = np.sum(input_x_positions != -1234567890)
        input_positions = np.zeros((num_input_pos, 2))
        idx = 0
        for row_idx, row in enumerate(input_x_positions):
            for col_idx, x_val in enumerate(row):
                if x_val != -1234567890:
                    input_positions[idx, 0] = x_val
                    input_positions[idx, 1] = input_y_positions[row_idx]
                    idx += 1

        return input_positions, horizontal_thread_y_pos
    
def get_closest_input_position(input_positions, input_x_pos, input_y_pos, horizontal_thread_y_pos, thread_length, dx):
    distances = np.linalg.norm(input_positions - np.array([input_x_pos, input_y_pos]), axis=1)
    closest_idx = np.argmin(distances)
    closest_position = input_positions[closest_idx]

    vib_thread_idx_list = [np.where(horizontal_thread_y_pos == closest_position[1])[0][0]]
    node_idx_list = [int((closest_position[0] + thread_length/2 - network_origin[0])/dx)]
    return vib_thread_idx_list, node_idx_list

if __name__ == '__main__':

    '''Optimization Parameters from Simulation'''
    network_origin = np.zeros((3,)) # network_origin is the center of the network
    num_horizontal_threads = 2
    num_vertical_threads = 2
    num_inputs = 1
    num_outputs = (num_horizontal_threads*num_vertical_threads) + (num_horizontal_threads + 1) * num_vertical_threads + (num_vertical_threads + 1) * num_horizontal_threads

    thread_length = 500e-3, # 1 m --> 1e3 mm
    thread_diameter = 2e-3, # 1 m --> 1e3 mm
    dx = 10e-3, # 1 m --> 1e3 mm

    input_positions, horizontal_thread_y_pos = get_input_node_positions(num_vertical_threads, num_horizontal_threads, thread_length, dx)

    input_x_pos = 0
    input_y_pos = 0

    vib_thread_idx_list, node_idx_list = get_closest_input_position(input_positions, input_x_pos, input_y_pos, horizontal_thread_y_pos, thread_length, dx)

    scaling_type = "mm_g_s"
    params = {
        'num_horizontal_threads': num_horizontal_threads,
        'num_vertical_threads': num_vertical_threads,
        'network_origin': network_origin, # network_origin is the center of the network

        'thread_length': thread_length, # 1 m --> 1e3 mm
        'thread_diameter': thread_diameter, # 1 m --> 1e3 mm
        'dx': dx, # 1 m --> 1e3 mm

        'youngs_modulus': 10e6, # 1 Pa = kg /m/s2 --> 1 g/mm/s2 --> 1e-3 mg/mm/ms2
        'density': 1e3, # 1 kg / mm3 --> 1e-6 g/mm3 --> 1e-3 mg/mm3

        'tension_force': 1e-2, # 1 N = kg m/s2 --> 1e6 g mm/s2 --> 1e3 mg mm /ms2  
        'point_force_mag': -5e-3, #-0.2 # 1 N = kg m/s2 --> 1e6 g mm/s2 --> 1e3 mg mm /ms2 
        'SPREAD_PF': True, # whether the force should be a gaussian spread across 5 nodes or just applied at a single point
        'TYPE_PF': "spline", # type of force to be applied 
        'sample_freq_pf': 5, # Sampling frequency for random point force
        'vib_thread_idx_list': vib_thread_idx_list,
        'node_idx_list': node_idx_list,

        'damping_constant': 10, 
        'filter_order': 6,

        'k': 1e9, # translational stiffness of connection
        'kt': 1e9, # rotational stiffness of connection
        'nu': 0.0, # translational damping of connection

        'duration': 10, # 1 s --> 1e3 ms
        'sim_dt': 5e-6, # simulation timestep

        'rendering_fps': 250,
        
        'STOP_AT_NAN': True,
        'SAVE': True,
        'VIDEO': True,

        'scaling_type': scaling_type,
        'loc': './Simulations/SAGE_optimization/', #'./SMASIS_constant_force/',
        'file_type': 'npz'
    }

    print(params)

    params = unit_scaling.scale(params=params, scaling_type=scaling_type)

    params_list = [params]

    ''' Launching the simulation'''
    mp_handler(params_list)