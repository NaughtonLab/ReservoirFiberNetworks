import os
import argparse
import multiprocessing
import numpy as np

from fiber_simulation_main import fiber_simulation
from utils.unit_scaling import unit_scaling
from utils import save_results as sr

'''
Configures and launches fiber_simulation, either the single default simulation below or one
simulation of a SAGE study (replaces the per-study fiber_config_<study>.py scripts).

    python fiber_config_main.py                                        # default configuration
    python fiber_config_main.py --case_name GS_Force_Spacing --idx 17  # row 17 of the study grid
    python fiber_config_main.py --case_name GS_Tension_Spacing --idx 3 --grid_file tension_spacing_force_grid.npz
    python fiber_config_main.py --case_name Output_Reduction --num_threads 6 --force 0.2
    python fiber_config_main.py --case_name GS_Tension_Spacing --idx 0 --dry_run   # print params only

Case names are the keys of utils.save_results.STUDIES (shared with fiber_network_evaluations.py)
plus Output_Reduction, which has no grid. Each study starts from default_params() and overrides
the network geometry, point force, duration and output folder from one grid row (see SIM_CASES).
All parameters are in SI units until unit_scaling.scale converts them.
'''

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

def default_params():
    '''The default single simulation. The SAGE studies share every value not set by their case.'''
    return {
        'num_horizontal_threads': 2,
        'num_vertical_threads': 2,
        'network_origin': np.zeros((3,)), # network_origin is the center of the network

        'thread_length': 500e-3, # 1 m --> 1e3 mm
        'thread_diameter': 2e-3, # 1 m --> 1e3 mm
        'dx': 10e-3, # 1 m --> 1e3 mm

        'youngs_modulus': 100e6, # 1 Pa = kg /m/s2 --> 1 g/mm/s2 --> 1e-3 mg/mm/ms2
        'density': 1e3, # 1 kg / mm3 --> 1e-6 g/mm3 --> 1e-3 mg/mm3

        'tension_force': 1e-2, # 1 N = kg m/s2 --> 1e6 g mm/s2 --> 1e3 mg mm /ms2
        'point_force_mag': -5e-3, #-0.2 # 1 N = kg m/s2 --> 1e6 g mm/s2 --> 1e3 mg mm /ms2
        'SPREAD_PF': True, # whether the force should be a gaussian spread across 5 nodes or just applied at a single point
        'TYPE_PF': "spline", # type of force to be applied
        'sample_freq': 5, # Sampling frequency for random point force

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

        'scaling_type': "mm_g_s",
        'loc': './Simulations/SMASIS/',
        'file_type': 'npz'
    }

def element_length(thread_length, spacing, fine_20_30mm):
    '''
    dx of the SAGE studies: 10 mm with at least 50 elements per thread. With fine_20_30mm,
    spacings of exactly 20 and 30 mm (spacing in m) use 5 mm.
    '''
    dx = 10e-3
    n_elem = np.rint(thread_length/dx).astype(int)
    if n_elem < 50:
        n_elem = 50
        dx = thread_length/n_elem

    if dx > 10e-3:
        dx = 10e-3

    if fine_20_30mm and (spacing*1e3 == 20.0 or spacing*1e3 == 30.0):
        dx = 5e-3
    return dx

'''
Grid row -> parameter overrides (SI). Each one repeats the arithmetic of the original
fiber_config_<study>.py so the simulations are identical. The grids store spacing in different
units, see utils.save_results.row_to_params.
'''
def force_spacing_params(row):
    # [force (N), spacing (m)], 4x4
    point_force_mag, spacing = row[0], row[1]
    num_threads = 4
    thread_length = spacing * (num_threads+1)
    return dict(num_horizontal_threads=num_threads, num_vertical_threads=num_threads,
                thread_length=thread_length, dx=element_length(thread_length, spacing, True),
                spacing=spacing, point_force_mag=-point_force_mag)

def force_sweep_params(row):
    # force (N), 4x4 at 70 mm spacing
    point_force_mag, spacing = row, 70e-3
    num_threads = 4
    thread_length = spacing * (num_threads+1)
    return dict(num_horizontal_threads=num_threads, num_vertical_threads=num_threads,
                thread_length=thread_length, dx=element_length(thread_length, spacing, False),
                spacing=spacing, point_force_mag=-point_force_mag)

def tension_spacing_params(row):
    # [tension (N), spacing (mm), force (N)], 4x4
    tension_force, spacing, point_force_mag = row[0], row[1]*1e-3, row[2]
    num_threads = 4
    thread_length = spacing * (num_threads+1)
    return dict(num_horizontal_threads=num_threads, num_vertical_threads=num_threads,
                thread_length=thread_length, dx=element_length(thread_length, spacing, False),
                spacing=spacing, tension_force=tension_force, point_force_mag=-point_force_mag)

def thread_spacing_params(row):
    # [num_threads, spacing (m), force (N)]
    num_threads, spacing, point_force_mag = int(row[0]), row[1], row[2]
    thread_length = spacing * (num_threads+1)
    return dict(num_horizontal_threads=num_threads, num_vertical_threads=num_threads,
                thread_length=thread_length, dx=element_length(thread_length, spacing, True),
                spacing=spacing, point_force_mag=-point_force_mag)

def increasing_density_params(row):
    # [num_threads, spacing (mm), force (N)], thread length fixed at 500 mm.
    # spacing is passed in mm as in the original runs: it only enters the file name
    # (spacing1.6667e+02m...), which utils.save_results.get_sim_name expects.
    num_threads, spacing, point_force_mag = int(row[0]), row[1], row[2]
    thread_length = 500e-3
    return dict(num_horizontal_threads=num_threads, num_vertical_threads=num_threads,
                thread_length=thread_length, dx=element_length(thread_length, spacing, True),
                spacing=spacing, point_force_mag=-point_force_mag)

def output_reduction_params(num_threads, force):
    # n x n network, thread length 500 mm, dx 10 mm; spacing and n_file are left to the simulator defaults
    return dict(num_horizontal_threads=num_threads, num_vertical_threads=num_threads,
                thread_length=500e-3, dx=10e-3, point_force_mag=-force)

# folder / data_subdir / grid_files default to utils.save_results.STUDIES[case].
SIM_CASES = {
    "GS_Force_Spacing": dict(row_params=force_spacing_params, duration=100),
    "Force_Sweep": dict(row_params=force_sweep_params, duration=50),
    "GS_Tension_Spacing": dict(row_params=tension_spacing_params, duration=100),
    "GS_Thread_Spacing_NL": dict(row_params=thread_spacing_params, duration=50),
    "GS_Thread_Spacing_MC": dict(row_params=thread_spacing_params, duration=50),
    "Increasing_Density": dict(row_params=increasing_density_params, duration=100),
    "Output_Reduction": dict(folder='Simulations/SAGE/OutputReduction', data_subdir='Data', duration=100),
}

def case_setting(case_name, key):
    sim_case = SIM_CASES[case_name]
    if key in sim_case:
        return sim_case[key]
    return sr.get_study(case_name)[key]

def resolve_grid_file(case_name, file):
    '''A path as given, or else a file name inside the study folder.'''
    if os.path.exists(file):
        return file
    return os.path.join(sr.REPO_ROOT, case_setting(case_name, 'folder'), file)

def load_grid(case_name, grid_files=None, grid_key=None):
    '''
    Rows of the study grid. Several files are concatenated in order, so Force-Spacing rows >=112
    come from force_spacing_grid_2.npz. grid_files / grid_key override the study's files; a file
    holding a single array is read whatever its key.
    '''
    if grid_files is None:
        files = [(resolve_grid_file(case_name, f), key) for f, key in case_setting(case_name, 'grid_files')]
    else:
        default_key = case_setting(case_name, 'grid_files')[0][1]
        files = [(resolve_grid_file(case_name, f), grid_key or default_key) for f in grid_files]

    grids = []
    for file, key in files:
        data = np.load(file, allow_pickle=True)
        if key not in data.files and len(data.files) == 1:
            key = data.files[0]
        grids.append(data[key])
    return np.concatenate(grids, axis=0)

def build_params(case_name=None, idx=None, grid_files=None, grid_key=None, num_threads=None, force=None,
                 output_dir=None, duration=None, video=None):
    '''Parameters in SI units. Without case_name: the default configuration.'''
    params = default_params()

    if case_name is not None:
        data_dir = os.path.normpath(os.path.join(sr.REPO_ROOT, case_setting(case_name, 'folder'), case_setting(case_name, 'data_subdir')))
        params.update(VIDEO=False, duration=case_setting(case_name, 'duration'), loc=os.path.join(data_dir, ''))
        if case_name == "Output_Reduction":
            params.update(output_reduction_params(num_threads, force))
        else:
            grid = load_grid(case_name, grid_files, grid_key)
            if not 0 <= idx < len(grid):
                raise IndexError(f"idx {idx} is outside the {case_name} grid (0-{len(grid)-1})")
            params.update(SIM_CASES[case_name]['row_params'](grid[idx]))
            params['n_file'] = idx

    if output_dir is not None:
        params['loc'] = os.path.join(output_dir, '')
    if duration is not None:
        params['duration'] = duration
    if video is not None:
        params['VIDEO'] = video
    return params

def describe(case_name, idx, params):
    n_elem = np.rint(params['thread_length']/params['dx']).astype(int)
    text = (f"{case_name or 'default'}" + (f" idx {idx}" if idx is not None else "") +
            f": {params['num_horizontal_threads']}x{params['num_vertical_threads']}, length {params['thread_length']*1e3:g} mm, "
            f"dx {params['dx']*1e3:g} mm, n_elem {n_elem}, tension {params['tension_force']:g} N, "
            f"force {params['point_force_mag']:g} N, duration {params['duration']:g} s")
    if 'spacing' in params:
        text += f", spacing {params['spacing']:g} (grid units)"
    return text + f" -> {params['loc']}"

def parse_args():
    parser = argparse.ArgumentParser(description="Configure and launch one fiber network simulation: the default "
                                                 "configuration, or one simulation of a SAGE study.")
    parser.add_argument('--case_name', '--case', choices=list(SIM_CASES),
                        help='study to run; omit for the default configuration')
    parser.add_argument('--idx', '--grid_idx', '--list_idx', dest='idx', type=int,
                        help='row of the study grid / sweep list (also the n_file suffix of the output)')
    parser.add_argument('--grid_file', nargs='+',
                        help='grid / sweep .npz file(s) replacing the study default, concatenated in order; '
                             'a bare file name is looked up in the study folder')
    parser.add_argument('--grid_key', help="array name inside --grid_file (default: the study's, e.g. 'grid' or 'sweep')")
    parser.add_argument('--num_threads', type=int, help='Output_Reduction: threads per direction')
    parser.add_argument('--force', type=float, help='Output_Reduction: point force magnitude in N')
    parser.add_argument('--output_dir', help='folder for the simulation output (must exist)')
    parser.add_argument('--duration', type=float, help='simulated time in s')
    parser.add_argument('--video', action=argparse.BooleanOptionalAction, default=None, help='render a video')
    parser.add_argument('--dry_run', action='store_true', help='print the parameters without simulating')
    args = parser.parse_args()

    if args.case_name == "Output_Reduction":
        if args.num_threads is None or args.force is None:
            parser.error("Output_Reduction needs --num_threads and --force")
    elif args.num_threads is not None or args.force is not None:
        parser.error("--num_threads and --force are only used by Output_Reduction")
    if args.case_name not in (None, "Output_Reduction"):
        if args.idx is None:
            parser.error(f"{args.case_name} needs --idx")
    elif args.idx is not None or args.grid_file or args.grid_key:
        parser.error("--idx, --grid_file and --grid_key need a study with a grid")
    return args

if __name__ == '__main__':
    args = parse_args()

    params = build_params(case_name=args.case_name, idx=args.idx, grid_files=args.grid_file, grid_key=args.grid_key,
                          num_threads=args.num_threads, force=args.force,
                          output_dir=args.output_dir, duration=args.duration, video=args.video)

    print(describe(args.case_name, args.idx, params))
    print(params)
    if args.dry_run:
        raise SystemExit

    params = unit_scaling.scale(params=params, scaling_type=params['scaling_type'])

    params_list = [params]

    ''' Launching the simulation'''
    # mp_handler(params_list)
    sim = fiber_simulation(**params)
    sim.launch_sim()
