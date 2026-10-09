import numpy as np
from scipy.interpolate import CubicSpline

'''
Random spline input used for the 'spline' point force. fiber_simulation_main.launch_sim builds the
force with this function and utils/extract_sim_data.load_simulation_data(regenerate_ip=True) rebuilds
the same input with it, so the knot times and random values always match.
Only numpy/scipy here, so the evaluation code does not need PyElastica.
'''

def generate_spline_inputs(duration, sample_freq, seed_value, n_splines=1):
    '''
    Returns n_splines CubicSplines with knots every 1/sample_freq time units on [0, ceil(duration)],
    values uniform in [-1, 1] and the first knot pinned to 0. The global numpy RNG is seeded with
    seed_value and the splines draw their values one after the other (one per stimulated thread).
    duration and sample_freq must be in the simulation's time units (after unit_scaling).
    '''
    np.random.seed(seed_value)

    sample_time = np.ceil(duration).astype(int)
    x_sample = np.linspace(0, sample_time, sample_time*sample_freq + 1)

    spline_list = []
    for _ in range(n_splines):
        y_sample = np.random.uniform(-1, 1, size=sample_time*sample_freq+1)
        y_sample[0] = 0.0
        spline_list.append(CubicSpline(x_sample, y_sample))

    return spline_list
