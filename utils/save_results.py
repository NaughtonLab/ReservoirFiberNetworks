import os
import json
import datetime
import numpy as np
import pandas as pd

'''
Study bookkeeping for fiber_network_evaluations.py: where each study lives, how a row of its
parameter grid maps to physical parameters (row_to_params), how the simulation file name is
rebuilt, and how evaluation results are written.

Outputs never overwrite the original CSV / _eval.npz files: every file name carries a tag built
from the evaluation configuration (see config_tag), and each CSV gets a .json sidecar with the
full configuration.
'''

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# grid_files: list of (file, npz key). Several files are concatenated in order, so the simulation
# index is the row index of the combined grid (e.g. Force-Spacing rows >=112 come from grid_2).
STUDIES = {
    "Increasing_Density": dict(folder='Simulations/SAGE/IncreasingDensity', data_subdir='Data',
                               grid_files=[('increasing_density_list.npz', 'grid')],
                               n_sims=6, csv_base='IncreasingDensityEvaluation'),
    "GS_Force_Spacing": dict(folder='Simulations/SAGE/GridSearch/ForceSpacing', data_subdir='Data',
                             grid_files=[('force_spacing_grid.npz', 'grid'), ('force_spacing_grid_2.npz', 'grid')],
                             n_sims=252, csv_base='GSEvaluation_force'),
    "Force_Sweep": dict(folder='Simulations/SAGE/GridSearch/ForceSweep', data_subdir='Data',
                        grid_files=[('forces_sweep.npz', 'sweep')],
                        n_sims=9, csv_base='ForceCliffSweep_evaluation'),
    "GS_Tension_Spacing": dict(folder='Simulations/SAGE/GridSearch/TensionSpacing', data_subdir='Data',
                               grid_files=[('tension_spacing_force_grid_new.npz', 'grid')],
                               n_sims=36, csv_base='GSEvaluation_tension'),
    "GS_Thread_Spacing_NL": dict(folder='Simulations/SAGE/GridSearch/ThreadSpacing', data_subdir='Data_NL',
                                 grid_files=[('thread_spacing_grid_NL.npz', 'grid')],
                                 n_sims=84, csv_base='GSEvaluation_NL'),
    "GS_Thread_Spacing_MC": dict(folder='Simulations/SAGE/GridSearch/ThreadSpacing', data_subdir='Data_MC',
                                 grid_files=[('thread_spacing_grid_MC.npz', 'grid')],
                                 n_sims=84, csv_base='GSEvaluation_MC'),
}

METRIC_COLUMNS = ['nonlinearity train', 'memory train', 'nonlinearity test', 'memory test',
                  'nonlinearity R2 train', 'memory R2 train', 'nonlinearity R2 test', 'memory R2 test']

def get_study(case_name):
    if case_name not in STUDIES:
        raise ValueError(f"Unknown case_name '{case_name}'. Choose from {list(STUDIES)}")
    return STUDIES[case_name]

def study_folder(case_name):
    return os.path.normpath(os.path.join(REPO_ROOT, get_study(case_name)['folder']))

def data_folder(case_name):
    return os.path.join(study_folder(case_name), get_study(case_name)['data_subdir'])

def load_grid(case_name):
    folder = study_folder(case_name)
    grids = [np.load(os.path.join(folder, f), allow_pickle=True)[key] for f, key in get_study(case_name)['grid_files']]
    return np.concatenate(grids, axis=0)

def row_to_params(case_name, grid_data):
    '''
    Physical parameters of one grid row, with spacing and length in mm. The grids do not store
    spacing in the same unit:
      Increasing_Density : [num_threads, spacing (mm), force (N)], thread length fixed at 500 mm
      GS_Force_Spacing   : [force (N), spacing (m)], 4 threads
      Force_Sweep        : force (N), 4 threads at 70 mm spacing
      GS_Tension_Spacing : [tension (N), spacing (mm), force (N)], 4 threads
      GS_Thread_Spacing_*: [num_threads, spacing (m), force (N)]
    Except Increasing_Density, thread_length = spacing * (num_threads + 1).
    '''
    match case_name:
        case "Increasing_Density":
            num_threads = int(grid_data[0])
            spacing_mm = float(grid_data[1])
            point_force_mag = float(grid_data[2])
            thread_length_mm = 500.0
            tension = None
        case "GS_Force_Spacing":
            num_threads = 4
            point_force_mag = float(grid_data[0])
            spacing_mm = float(grid_data[1])*1e3
            thread_length_mm = spacing_mm * (num_threads+1)
            tension = None
        case "Force_Sweep":
            num_threads = 4
            point_force_mag = float(grid_data)
            spacing_mm = 70.0
            thread_length_mm = spacing_mm * (num_threads+1)
            tension = None
        case "GS_Tension_Spacing":
            num_threads = 4
            tension = float(grid_data[0])
            spacing_mm = float(grid_data[1])
            point_force_mag = float(grid_data[2])
            thread_length_mm = spacing_mm * (num_threads+1)
        case "GS_Thread_Spacing_NL" | "GS_Thread_Spacing_MC":
            num_threads = int(grid_data[0])
            spacing_mm = float(grid_data[1])*1e3
            point_force_mag = float(grid_data[2])
            thread_length_mm = spacing_mm * (num_threads+1)
            tension = None
        case _:
            raise ValueError(f"Unknown case_name '{case_name}'")

    params = {'num_threads': num_threads, 'spacing(mm)': spacing_mm, 'length(mm)': thread_length_mm}
    if tension is not None:
        params['tension(N)'] = tension
    params['force_mag(N)'] = point_force_mag
    return params

# Tension of every study except GS_Tension_Spacing (fiber_config_main.default_params()['tension_force'])
DEFAULT_TENSION_N = 1e-2

def get_sim_name(case_name, params, idx, fps, stepskip, legacy=False):
    '''
    Rebuilds the simulation file name (without extension) that fiber_simulation.file_name() writes
    for a study run by fiber_config_main.py:
        {n}by{n}rods_spacing{s}m_TF{tension}N_PF{-force}Nspline_fps{fps}_stepskip{stepskip}_{idx}
    The spacing number is in mm for Increasing_Density (as the original runs passed it) and in m
    otherwise; the tension is DEFAULT_TENSION_N unless the study sets it.
    legacy=True gives the names of the original SAGE grid-search files (on ARC), which have no TF
    block, except Increasing_Density whose names are the same in both formats.
    '''
    num_threads = params['num_threads']
    force = params['force_mag(N)']
    tension = params.get('tension(N)', DEFAULT_TENSION_N)
    if case_name == "Increasing_Density":
        spacing = params['spacing(mm)']
    else:
        spacing = params['spacing(mm)']*1e-3

    tension_block = '' if (legacy and case_name != "Increasing_Density") else f'_TF{tension:.0e}N'
    suffix = f"spacing{spacing:.4e}m{tension_block}_PF{-force:.0e}Nspline_fps{fps}_stepskip{stepskip}"
    return f'{num_threads}by{num_threads}rods_{suffix}_{idx}'

def find_sim_file(case_name, params, idx, fps, stepskip, folder=None):
    '''
    Path (without .npz) of the simulation file in the study's data folder: the current name if
    that file exists, else the legacy name if that one exists, else the current name.
    '''
    folder = folder or data_folder(case_name)
    paths = [os.path.join(folder, get_sim_name(case_name, params, idx, fps, stepskip, legacy=legacy)) for legacy in (False, True)]
    for path in paths:
        if os.path.exists(f'{path}.npz'):
            return path
    return paths[0]

def __fmt__(x):
    return f'{x:.0e}'

def config_tag(cfg, alpha=None):
    '''
    Short description of the evaluation configuration used in output file names, e.g.
    Rid_a1e-02_split0.25_L10_D250 or Rid_a1e-04to1e+02x7_cv10-kf-tss_L10_D250.
    With alpha given, the tag describes that single alpha (used for the per-alpha npz files).
    '''
    abbr = {'KFold': 'kf', 'TimeSeriesSplit': 'tss'}
    parts = [cfg.regressor]
    if cfg.regressor == "Rid":
        alphas = [alpha] if alpha is not None else list(cfg.alphas)
        if len(alphas) == 1:
            parts.append(f'a{__fmt__(alphas[0])}')
        else:
            parts.append(f'a{__fmt__(min(alphas))}to{__fmt__(max(alphas))}x{len(alphas)}')
    if cfg.CV:
        parts.append(f'cv{cfg.n_splits}-{abbr.get(cfg.type_CV_nonlin, cfg.type_CV_nonlin)}-{abbr.get(cfg.type_CV_mem, cfg.type_CV_mem)}')
    else:
        parts.append(f'split{cfg.test_size:g}')
    parts.append(f'L{cfg.leg_max_order}')
    parts.append(f'D{cfg.max_timesteps_back}')
    if not cfg.regenerate_ip:
        parts.append('ipsaved')
    else:
        if cfg.seed_value != 1234:
            parts.append(f'seed{cfg.seed_value}')
        if cfg.sample_freq != 5:
            parts.append(f'sf{cfg.sample_freq}')
        if cfg.duration is not None:
            parts.append(f'dur{cfg.duration:g}')
    if cfg.tag:
        parts.append(cfg.tag)
    return '_'.join(parts)

def results_csv_path(case_name, cfg):
    return os.path.join(study_folder(case_name), f"{get_study(case_name)['csv_base']}__{config_tag(cfg)}.csv")

def eval_npz_path(case_name, cfg, idx, alpha):
    return os.path.join(data_folder(case_name), f"{idx}_eval__{config_tag(cfg, alpha)}.npz")

def summary_row(idx, params, alpha, results):
    '''One CSV row: grid parameters, alpha, and the mean capacity and mean R2 of every test.'''
    row = {'idx': idx, **params, 'alpha': alpha}
    for test in ('nonlinearity', 'memory'):
        cap_train, cap_test, R2_train, R2_test = results[test]
        row[f'{test} train'] = np.mean(cap_train)
        row[f'{test} test'] = np.mean(cap_test)
        row[f'{test} R2 train'] = np.mean(R2_train)
        row[f'{test} R2 test'] = np.mean(R2_test)
    return row

def save_eval_npz(path, results, input_data, output_data, time_data, cfg_dict, save_io_data=True):
    '''Same keys as the original _eval.npz files, plus the evaluation config.'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = dict(nonlinearity=np.asarray(results['nonlinearity']),
                  memory=np.asarray(results['memory']),
                  config=json.dumps(cfg_dict))
    if results.get('heatmap') is not None:
        arrays['heatmap'] = np.asarray(results['heatmap'])
    if save_io_data:
        arrays.update(input_data=input_data, output_data=output_data, time_data=time_data)
    np.savez(path, **arrays)

def update_results_csv(csv_path, rows, cfg_dict):
    '''
    Adds rows to the CSV for this configuration, replacing rows with the same (idx, alpha), so a
    run can be split into several idx ranges. Also writes <csv>.json with the configuration.
    Not safe for several processes writing the same CSV at once.
    '''
    new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        keys = set(zip(new['idx'], new['alpha']))
        old = old[[(i, a) not in keys for i, a in zip(old['idx'], old['alpha'])]]
        new = pd.concat([old, new], ignore_index=True)
    columns = [c for c in new.columns if c not in METRIC_COLUMNS] + METRIC_COLUMNS
    new = new[columns].sort_values(['idx', 'alpha']).reset_index(drop=True)
    new.to_csv(csv_path, index=False)

    meta = dict(cfg_dict, csv=os.path.basename(csv_path), updated=datetime.datetime.now().isoformat(timespec='seconds'))
    with open(os.path.splitext(csv_path)[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=2)
