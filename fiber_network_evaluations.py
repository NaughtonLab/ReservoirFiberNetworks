import os
import json
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.extract_sim_data import load_simulation_data
from utils.evaluate import nonlinearity_testing, memory_testing, nonlinearity_memory_matrix
from utils import save_results as sr

'''
Evaluates the nonlinearity and memory capacities of saved simulations for any SAGE study
(replaces fiber_network_evaluations_SAGE.py, fiber_network_evaluations_SAGE_alphaCV.py and the
per-study get_*_eval*.py scripts).

Settings come from EvalConfig. Its defaults are overridden by, in order:
  1. the CONFIG dict in __main__ (or a JSON file passed with --config),
  2. command line arguments, e.g.
       python fiber_network_evaluations.py --case_name GS_Force_Spacing --idx 0-111
       python fiber_network_evaluations.py --case_name Increasing_Density --CV --alphas 1e-4 1e-3 1e-2 --no-compute_matrix

For every simulation index and alpha it writes <idx>_eval__<tag>.npz in the study's Data folder,
and one row per (idx, alpha) in <csv_base>__<tag>.csv in the study folder (both capacity and R2
means). The tag describes the configuration (see utils.save_results.config_tag), so existing
outputs are never overwritten.
'''

@dataclass
class EvalConfig:
    case_name: str = "GS_Thread_Spacing_NL"     # key of utils.save_results.STUDIES
    idx_list: Optional[list] = None             # None -> every simulation of the study

    regressor: str = "Rid"                      # "Rid" or "Lin"
    alphas: list = field(default_factory=lambda: [1e-2])
    test_size: float = 0.25                     # single split only (and the matrix)
    leg_max_order: int = 10
    fps: int = 250
    max_time_back_seconds: float = 1

    CV: bool = False                            # True -> utils/evaluate_cv.py for nonlinearity and memory
    type_CV_nonlin: str = "KFold"
    type_CV_mem: str = "TimeSeriesSplit"
    n_splits: int = 10
    compute_matrix: bool = True                 # nonlinearity-memory matrix (always a single split)

    source: str = "raw"                         # "raw": simulation .npz, "cached": <idx>_eval.npz from an earlier run
    regenerate_ip: bool = True
    seed_value: int = 1234
    sample_freq: int = 5                        # spline knots per time unit, as in the simulation config
    duration: Optional[float] = None            # simulated duration; None -> last saved time rounded
    stepskip: int = 800
    save_io_data: bool = True                   # store input/output/time data in every eval npz
    tag: str = ""                               # extra text appended to output file names
    save_predictions_dir: Optional[str] = None  # folder for y_test / y_test_pred npz files (single split only); None -> not saved

    @property
    def max_timesteps_back(self):
        return int(np.rint(self.fps*self.max_time_back_seconds))

    def to_dict(self):
        return dataclasses.asdict(self)

def parse_idx(text):
    '''"0-83" -> 0..83, "0,4,10-12" -> [0, 4, 10, 11, 12]'''
    idx = []
    for part in text.split(','):
        if '-' in part:
            start, stop = part.split('-')
            idx.extend(range(int(start), int(stop)+1))
        else:
            idx.append(int(part))
    return idx

def parse_args():
    # Every default is SUPPRESS, so only the arguments actually given override the config.
    parser = argparse.ArgumentParser(description="Evaluate nonlinearity and memory capacities of SAGE simulations. "
                                                 "Arguments override CONFIG in __main__ and EvalConfig defaults.",
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('--config', help='JSON file with EvalConfig fields')
    parser.add_argument('--case_name', choices=list(sr.STUDIES))
    parser.add_argument('--idx', dest='idx_list', type=parse_idx, help='e.g. 0-83 or 0,4,10-12')
    parser.add_argument('--regressor', choices=['Rid', 'Lin'])
    parser.add_argument('--alphas', type=float, nargs='+')
    parser.add_argument('--test_size', type=float)
    parser.add_argument('--leg_max_order', type=int)
    parser.add_argument('--fps', type=int)
    parser.add_argument('--max_time_back_seconds', type=float)
    parser.add_argument('--CV', action=argparse.BooleanOptionalAction)
    parser.add_argument('--type_CV_nonlin', choices=['KFold', 'TimeSeriesSplit'])
    parser.add_argument('--type_CV_mem', choices=['KFold', 'TimeSeriesSplit'])
    parser.add_argument('--n_splits', type=int)
    parser.add_argument('--compute_matrix', action=argparse.BooleanOptionalAction)
    parser.add_argument('--source', choices=['raw', 'cached'])
    parser.add_argument('--regenerate_ip', action=argparse.BooleanOptionalAction)
    parser.add_argument('--seed_value', type=int)
    parser.add_argument('--sample_freq', type=int)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--stepskip', type=int)
    parser.add_argument('--save_io_data', action=argparse.BooleanOptionalAction)
    parser.add_argument('--tag')
    parser.add_argument('--save_predictions_dir', help='save y_test / y_test_pred of every test in this folder')
    return vars(parser.parse_args())

def build_config(config_dict=None, cli_args=None):
    '''EvalConfig defaults <- config_dict <- --config JSON file <- remaining CLI args. Unknown keys raise.'''
    settings = dict(config_dict or {})
    cli_args = dict(cli_args or {})
    config_file = cli_args.pop('config', None)
    if config_file:
        with open(config_file) as f:
            settings.update(json.load(f))
    settings.update(cli_args)
    return EvalConfig(**settings)

def load_data(cfg, params, idx):
    '''Returns (input, output, time) or (None, None, None) when the data is missing.'''
    path = sr.data_folder(cfg.case_name)
    if cfg.source == "cached":
        cached = os.path.join(path, f"{idx}_eval.npz")
        if not os.path.exists(cached):
            print('File does not exist', cached)
            return None, None, None
        data = np.load(cached, allow_pickle=True)
        return data['input_data'], data['output_data'], data['time_data']

    # Current file name (fiber_config_main.py runs), falling back to the legacy name of the original ARC files
    sim_file = sr.find_sim_file(cfg.case_name, params, idx, cfg.fps, cfg.stepskip, folder=path)
    sim_ip_data, sim_op_data, sim_time_data = load_simulation_data(file_path = sim_file,
                                                                    file_type = 'npz',
                                                                    start = 0,
                                                                    num_horizontal_threads = params['num_threads'],
                                                                    num_vertical_threads = params['num_threads'],
                                                                    step = 1,
                                                                    regenerate_ip = cfg.regenerate_ip,
                                                                    seed_value = cfg.seed_value,
                                                                    sample_freq = cfg.sample_freq,
                                                                    duration = cfg.duration)
    if len(sim_ip_data) == 0:   # loading raised an exception
        return None, None, None
    return sim_ip_data[0], sim_op_data[0], sim_time_data[0]

def evaluate(input_data, output_data, cfg, alpha, idx):
    '''results[test] = [capacity_train, capacity_test, R2_train, R2_test]'''
    pred_paths = {'nonlinearity': None, 'memory': None}
    if cfg.save_predictions_dir:
        tag = sr.config_tag(cfg, alpha)
        pred_paths = {test: os.path.join(cfg.save_predictions_dir, f"{idx}_predictions_{test}__{tag}.npz") for test in pred_paths}

    results = {}
    results['nonlinearity'] = nonlinearity_testing(input_data, output_data, cfg.leg_max_order, cfg.regressor, cfg.test_size, alpha,
                                                   CV=cfg.CV, type_CV=cfg.type_CV_nonlin, n_splits=cfg.n_splits,
                                                   save_predictions_path=pred_paths['nonlinearity'])
    results['memory'] = memory_testing(input_data, output_data, cfg.max_timesteps_back, cfg.regressor, cfg.test_size, alpha,
                                       CV=cfg.CV, type_CV=cfg.type_CV_mem, n_splits=cfg.n_splits,
                                       save_predictions_path=pred_paths['memory'])
    if cfg.compute_matrix:
        results['heatmap'] = nonlinearity_memory_matrix(input_data, output_data, cfg.leg_max_order, cfg.max_timesteps_back,
                                                        cfg.regressor, cfg.test_size, alpha)
    return results

def nan_results(cfg):
    results = {'nonlinearity': np.full((4, cfg.leg_max_order), np.nan),
               'memory': np.full((4, cfg.max_timesteps_back+1), np.nan)}
    if cfg.compute_matrix:
        results['heatmap'] = np.full((4, cfg.leg_max_order, cfg.max_timesteps_back+1), np.nan)
    return results

def run(cfg):
    grid = sr.load_grid(cfg.case_name)
    idx_list = cfg.idx_list if cfg.idx_list is not None else list(range(sr.get_study(cfg.case_name)['n_sims']))
    csv_path = sr.results_csv_path(cfg.case_name, cfg)
    cfg_dict = cfg.to_dict()
    print(f"Evaluating {cfg.case_name}, {len(idx_list)} simulations -> {csv_path}")

    for idx in idx_list:
        params = sr.row_to_params(cfg.case_name, grid[idx])
        input_data, output_data, time_data = load_data(cfg, params, idx)

        valid = input_data is not None and not np.isnan(input_data).any()
        if valid:
            input_data = -1 + (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data)) * (1 - (-1))
            print(idx, input_data.shape, output_data.shape)
        else:
            print(idx, "no valid data, writing NaN results")

        rows = []
        for alpha in cfg.alphas:
            results = evaluate(input_data, output_data, cfg, alpha, idx) if valid else nan_results(cfg)
            sr.save_eval_npz(sr.eval_npz_path(cfg.case_name, cfg, idx, alpha), results,
                             input_data, output_data, time_data, cfg_dict, save_io_data=cfg.save_io_data and valid)
            rows.append(sr.summary_row(idx, params, alpha, results))

        # Written after every simulation so a crashed or partial run keeps what it finished.
        sr.update_results_csv(csv_path, rows, cfg_dict)
        print(idx, "eval done.")

    print(f"Results saved in {csv_path}")


if __name__ == "__main__":

    CONFIG = dict(
        case_name = "GS_Thread_Spacing_NL",
    )

    ### Alpha sweep with cross validation (was fiber_network_evaluations_SAGE_alphaCV.py)
    # CONFIG = dict(
    #     case_name = "Increasing_Density",
    #     alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
    #     CV = True, type_CV_nonlin = 'KFold', type_CV_mem = 'TimeSeriesSplit', n_splits = 10,
    #     compute_matrix = False,
    # )

    run(build_config(CONFIG, parse_args()))
