import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from tests._helpers import SLOW, INC_DENSITY_FOLDER, INC_DENSITY_DATA, quiet

from utils import save_results as sr
import fiber_network_evaluations as driver

'''
Regression test against real results: the cached Increasing_Density files
(Simulations/SAGE/IncreasingDensity/Data/<idx>_eval*.npz and the study CSVs) were written by the
legacy scripts. Their input/output data is re-scored with the new code and compared with the
scores stored next to it. Skipped when the data files are not on this machine (they are not in git).

By default only idx 0 (2x2 network, 32 features, ~1 min). FIBER_SLOW_TESTS=1 runs all 6 networks,
the nonlinearity-memory heatmap and the full alpha sweep (much longer: idx 5 has 912 features).
'''

ATOL = 1e-9
IDX_LIST = list(range(6)) if SLOW else [0]
ALPHAS_CV = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2] if SLOW else [1e-2]

def cached(idx, suffix=''):
    return os.path.join(INC_DENSITY_DATA, f'{idx}_eval{suffix}.npz')

HAVE_DATA = os.path.exists(cached(0))


@unittest.skipUnless(HAVE_DATA, f'cached evaluation files not found in {INC_DENSITY_DATA}')
class TestRescoreCachedIncreasingDensity(unittest.TestCase):

    def load(self, idx):
        '''Input/output exactly as fiber_network_evaluations.run() prepares them (source="cached").'''
        d = np.load(cached(idx), allow_pickle=True)
        ip = d['input_data']
        ip = -1 + (ip - np.min(ip)) / (np.max(ip) - np.min(ip)) * (1 - (-1))
        return ip, d['output_data'], d

    def assert_close(self, new, old, what):
        new, old = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
        self.assertEqual(new.shape, old.shape, what)
        diff = np.nanmax(np.abs(new - old))
        self.assertLessEqual(diff, ATOL, f'{what}: max |new - cached| = {diff:.3e}')

    def test_single_split(self):
        cfg = driver.EvalConfig(case_name="Increasing_Density", compute_matrix=SLOW)
        for idx in IDX_LIST:
            with self.subTest(idx=idx):
                ip, op, d = self.load(idx)
                res = driver.evaluate(ip, op, cfg, 1e-2, idx)
                self.assert_close(res['nonlinearity'], d['nonlinearity'], 'nonlinearity')
                self.assert_close(res['memory'], d['memory'], 'memory')
                if SLOW:
                    self.assert_close(res['heatmap'], d['heatmap'], 'heatmap')

    def test_cross_validation(self):
        # _10fold: KFold for both tests; _10tscv: TimeSeriesSplit for both (get_IncDensity_data_eval_{kfold,tscv}.py)
        for suffix, type_CV in (('_10fold', 'KFold'), ('_10tscv', 'TimeSeriesSplit')):
            cfg = driver.EvalConfig(case_name="Increasing_Density", CV=True, n_splits=10, compute_matrix=False,
                                    type_CV_nonlin=type_CV, type_CV_mem=type_CV)
            for idx in IDX_LIST:
                if not os.path.exists(cached(idx, suffix)):
                    continue
                with self.subTest(file=suffix, idx=idx):
                    ip, op, _ = self.load(idx)
                    ref = np.load(cached(idx, suffix))
                    res = driver.evaluate(ip, op, cfg, 1e-2, idx)
                    self.assert_close(res['nonlinearity'], ref['nonlinearity'], 'nonlinearity')
                    self.assert_close(res['memory'], ref['memory'], 'memory')

    def test_alpha_sweep(self):
        # _10_alphaCV: KFold nonlinearity, TimeSeriesSplit memory, record (cap train/test nonlin, cap train/test mem)
        cfg = driver.EvalConfig(case_name="Increasing_Density", CV=True, n_splits=10, compute_matrix=False)
        for idx in IDX_LIST:
            if not os.path.exists(cached(idx, '_10_alphaCV')):
                continue
            ip, op, _ = self.load(idx)
            record = np.load(cached(idx, '_10_alphaCV'), allow_pickle=True)['npsavez_dict'].item()
            for alpha in ALPHAS_CV:
                with self.subTest(idx=idx, alpha=alpha):
                    res = driver.evaluate(ip, op, cfg, alpha, idx)
                    nl_train, nl_test, mem_train, mem_test = record[f'{alpha}']
                    self.assert_close(res['nonlinearity'][0], nl_train, 'nonlinearity train')
                    self.assert_close(res['nonlinearity'][1], nl_test, 'nonlinearity test')
                    self.assert_close(res['memory'][0], mem_train, 'memory train')
                    self.assert_close(res['memory'][1], mem_test, 'memory test')


@unittest.skipUnless(HAVE_DATA, f'cached evaluation files not found in {INC_DENSITY_DATA}')
class TestDriverReproducesStudyCSV(unittest.TestCase):
    '''fiber_network_evaluations.run() on cached data reproduces the rows of the legacy CSVs.'''

    def run_driver(self, **cfg_kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg = driver.build_config(dict(case_name="Increasing_Density", idx_list=IDX_LIST, source="cached",
                                       save_io_data=False, **cfg_kwargs))
        csv_path = os.path.join(tmp.name, 'results.csv')
        # read from the real Data folder, write only to the temp dir
        with mock.patch.object(sr, 'results_csv_path', lambda case, c: csv_path), \
             mock.patch.object(sr, 'eval_npz_path', lambda case, c, idx, alpha: os.path.join(tmp.name, f'{idx}_{alpha:g}.npz')), quiet():
            driver.run(cfg)
        return pd.read_csv(csv_path)

    def test_single_split_csv(self):
        legacy = pd.read_csv(os.path.join(INC_DENSITY_FOLDER, 'IncreasingDensityEvaluation.csv'))
        new = self.run_driver(compute_matrix=False)
        for idx in IDX_LIST:
            with self.subTest(idx=idx):
                old, row = legacy.iloc[idx], new[new['idx'] == idx].iloc[0]
                self.assertEqual(row['num_threads'], old['num_threads'])
                self.assertAlmostEqual(row['spacing(mm)'], old['spacing(mm)'], places=9)
                self.assertAlmostEqual(row['force_mag(N)'], old['force_mag(N)'], places=12)
                # legacy: 'nonlinearity'/'memory' = mean R2 test, '*_cap' = mean capacity test
                self.assertAlmostEqual(row['nonlinearity R2 test'], old['nonlinearity'], delta=ATOL)
                self.assertAlmostEqual(row['memory R2 test'], old['memory'], delta=ATOL)
                self.assertAlmostEqual(row['nonlinearity test'], old['nonlinearity_cap'], delta=ATOL)
                self.assertAlmostEqual(row['memory test'], old['memory_cap'], delta=ATOL)

    def test_alpha_sweep_csv(self):
        legacy = pd.read_csv(os.path.join(INC_DENSITY_FOLDER, 'IncreasingDensity_alpha_sweep_10CV.csv'))
        new = self.run_driver(alphas=ALPHAS_CV, CV=True, n_splits=10, compute_matrix=False)
        grid = sr.load_grid("Increasing_Density")
        for idx in IDX_LIST:
            n = int(grid[idx][0])
            for alpha in ALPHAS_CV:
                with self.subTest(idx=idx, alpha=alpha):
                    old = legacy[(legacy['num_threads'] == n) & np.isclose(legacy['alpha'], alpha)].iloc[0]
                    row = new[(new['idx'] == idx) & np.isclose(new['alpha'], alpha)].iloc[0]
                    for col in ('nonlinearity train', 'nonlinearity test', 'memory train', 'memory test'):
                        self.assertAlmostEqual(row[col], old[col], delta=ATOL, msg=col)


if __name__ == '__main__':
    unittest.main()
