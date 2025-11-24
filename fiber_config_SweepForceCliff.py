import os
import pickle
import argparse
import multiprocessing
import json
import numpy as np

from fiber_simulation_main import fiber_simulation
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

if __name__ == '__main__':

    parent_folder = "Simulations/SAGE/ForceCliff"

    sweep = np.load(f'{parent_folder}/forces_sweep_corrected.npz', allow_pickle=True)
    sweep = sweep['sweep']

    parser = argparse.ArgumentParser(description="Script that configures and launches fiber_simulation for a sweep of point forces to investigate the cliff in heatmap")

    parser.add_argument("--list_idx", required=True, type=int)
    args = parser.parse_args()
    list_idx = args.list_idx

    data = sweep[list_idx]
    
    point_force_mag = data
    spacing = 70e-3

    num_horizontal_threads = 4
    num_vertical_threads = num_horizontal_threads
    thread_length = spacing * (num_vertical_threads+1)

    dx = 10e-3
    n_elem = np.rint(thread_length/dx).astype(int)
    if n_elem < 50:
        n_elem = 50
        dx = thread_length/n_elem

    if dx > 10e-3:
        dx = 10e-3
        n_elem = np.rint(thread_length/dx).astype(int)

    print("list idx", list_idx, "force", point_force_mag, "spacing", spacing*1e3, "length", thread_length*1e3, "dx", dx*1e3, "n_elem", n_elem)

    scaling_type = "mm_g_s"
    params = {
        'num_horizontal_threads': num_horizontal_threads,
        'num_vertical_threads': num_vertical_threads,
        'network_origin': np.zeros((3,)), # network_origin is the center of the network

        'thread_length': thread_length, # 1 m --> 1e3 mm
        'thread_diameter': 2e-3, # 1 m --> 1e3 mm
        'dx': dx, # 1 m --> 1e3 mm
        'spacing': spacing,

        'youngs_modulus': 100e6, # 1 Pa = kg /m/s2 --> 1 g/mm/s2 --> 1e-3 mg/mm/ms2
        'density': 1e3, # 1 kg / mm3 --> 1e-6 g/mm3 --> 1e-3 mg/mm3

        'tension_force': 1e-2, # 1 N = kg m/s2 --> 1e6 g mm/s2 --> 1e3 mg mm /ms2  
        'point_force_mag': -point_force_mag, # 1 N = kg m/s2 --> 1e6 g mm/s2 --> 1e3 mg mm /ms2 
        'SPREAD_PF': True, # whether the force should be a gaussian spread across 5 nodes or just applied at a single point
        'TYPE_PF': "spline", # type of force to be applied 
        'sample_freq_pf': 5, # Sampling frequency for random point force

        'damping_constant': 10, 
        'filter_order': 6,

        'k': 1e9, # translational stiffness of connection
        'kt': 1e9, # rotational stiffness of connection
        'nu': 0.0, # translational damping of connection

        'duration': 50, # 1 s --> 1e3 ms
        'sim_dt': 5e-6, # simulation timestep

        'rendering_fps': 250,
        
        'STOP_AT_NAN': True,
        'SAVE': True,
        'VIDEO': False,

        'scaling_type': scaling_type,
        'loc': f'{parent_folder}/Data/',
        'file_type': 'npz',
        'n_file': list_idx
    }

    params = unit_scaling.scale(params=params, scaling_type=scaling_type)

    params_list = [params]

    ''' Launching the simulation'''
    # mp_handler(params_list)
    sim = fiber_simulation(**params)
    sim.launch_sim()