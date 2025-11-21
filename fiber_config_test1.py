import os
import pickle
import multiprocessing
import json
import numpy as np

from fiber_simulation_test1 import fiber_simulation
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
    scaling_type = "mm_g_s"
    params = {
        'num_horizontal_threads': 6,
        'num_vertical_threads': 6,
        'network_origin': np.zeros((3,)), # network_origin is the center of the network

        'thread_length': 500e-3, # 1 m --> 1e3 mm
        'thread_diameter': 2e-3, # 1 m --> 1e3 mm
        'dx': 10e-3, # 1 m --> 1e3 mm

        'youngs_modulus': 100e6, # 1 Pa = kg /m/s2 --> 1 g/mm/s2 --> 1e-3 mg/mm/ms2
        'density': 1e3, # 1 kg / mm3 --> 1e-6 g/mm3 --> 1e-3 mg/mm3

        'tension_force': 1e-2, # 1 N = kg m/s2 --> 1e6 g mm/s2 --> 1e3 mg mm /ms2  
        'point_force_mag': -0.2, # 1 N = kg m/s2 --> 1e6 g mm/s2 --> 1e3 mg mm /ms2 
        'SPREAD_PF': True, # whether the force should be a gaussian spread across 5 nodes or just applied at a single point
        'TYPE_PF': "spline", # type of force to be applied 
        'sample_freq_pf': 5, # Sampling frequency for random point force

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
        'loc': './Simulations/Tests/',
        'file_type': 'npz'
    }

    print(params)

    params = unit_scaling.scale(params=params, scaling_type=scaling_type)

    params_list = [params]

    ''' Launching the simulation'''
    # mp_handler(params_list)
    sim = fiber_simulation(**params)
    sim.launch_sim()