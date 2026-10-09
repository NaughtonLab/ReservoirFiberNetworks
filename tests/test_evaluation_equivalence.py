import os
import json
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from tests._helpers import (load_module, quiet, synthetic_reservoir, synthetic_rods_history)

from utils import evaluate as new_eval
from utils import evaluate_cv
from utils import extract_sim_data
from utils import save_results as sr
import fiber_network_evaluations as driver

'''
Old evaluation code vs new evaluation code on the same synthetic reservoir data (no data files
needed). "Old" means:
  - tests/reference/evaluate_legacy.py: utils/evaluate.py as committed before the refactor,
    used by fiber_network_evaluations_SAGE*.py;
  - the per-study scripts under Simulations/SAGE/**, which carry their own copies of the
    scoring / loading / preprocessing functions.
The refactor only moved code around, so scores must be bit-identical (assert_array_equal),
except where a legacy script is known to differ (ForceSweep: no clipping of train predictions).
'''

legacy_eval = load_module('tests/reference/evaluate_legacy.py', 'evaluate_legacy')

LEGACY_SINGLE_SPLIT_SCRIPTS = {
    'IncreasingDensity': 'Simulations/SAGE/IncreasingDensity/get_IncDensity_data_eval.py',
    'ForceSpacing': 'Simulations/SAGE/GridSearch/ForceSpacing/get_GSData_eval_force.py',
    'TensionSpacing': 'Simulations/SAGE/GridSearch/TensionSpacing/get_GSData_and_eval_tension.py',
    'ThreadSpacing_NL': 'Simulations/SAGE/GridSearch/ThreadSpacing/GSThreadSpacing_data_eval_NL.py',
    'ThreadSpacing_MC': 'Simulations/SAGE/GridSearch/ThreadSpacing/GSThreadSpacing_data_eval_MC.py',
}
LEGACY_FORCE_SWEEP = 'Simulations/SAGE/GridSearch/ForceSweep/get_SweepData_and_eval_cliff.py'
LEGACY_KFOLD = 'Simulations/SAGE/IncreasingDensity/get_IncDensity_data_eval_kfold.py'
LEGACY_TSCV = 'Simulations/SAGE/IncreasingDensity/get_IncDensity_data_eval_tscv.py'
LEGACY_ALPHA_CV = 'Simulations/SAGE/IncreasingDensity/get_IncDensity_data_eval_alphaCV.py'

L_MAX = 6          # Legendre orders (production: 10)
D_MAX = 30         # memory delays (production: 250)
N_SPLITS = 5       # CV folds (production: 10)
TEST_SIZE = 0.25
REGRESSORS = [("Lin", None), ("Rid", 1e-4), ("Rid", 1e-2), ("Rid", 1.0)]

def data():
    '''Fresh copies every call: __fit__ writes into y_train / y_test.'''
    return synthetic_reservoir(n_samples=3000, n_features=16, seed=0)

def assert_same(test, new, old, msg=''):
    test.assertEqual(len(new), len(old), msg)
    for k, (a, b) in enumerate(zip(new, old)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f'{msg} [output {k}]')


class TestSyntheticDataIsUseful(unittest.TestCase):
    '''Guards the other tests: the toy reservoir must give non-trivial scores and hit clipping.'''

    def test_capacities_are_non_trivial(self):
        ip, op = data()
        _, cap_test, _, _ = new_eval.nonlinearity_testing(ip, op, L_MAX, "Rid", TEST_SIZE, 1e-2)
        self.assertGreater(max(cap_test), 0.3)
        self.assertLess(min(cap_test), 0.99)
        _, mem_test, _, _ = new_eval.memory_testing(ip, op, D_MAX, "Rid", TEST_SIZE, 1e-2)
        self.assertGreater(max(mem_test), 0.3)

    def test_clipping_branch_is_exercised(self):
        from sklearn.linear_model import LinearRegression
        ip, op = data()
        hits = 0
        for n in range(1, L_MAX+1):
            y = new_eval.legendre(n)(ip)
            pred = LinearRegression().fit(op, y).predict(op)
            hits += int(np.sum(np.abs(pred) > 1))
        self.assertGreater(hits, 0, "synthetic data never produces |prediction| > 1")


class TestSingleSplitMatchesLegacyEvaluate(unittest.TestCase):
    '''New utils/evaluate.py vs the committed version (fiber_network_evaluations_SAGE.py path).'''

    def test_nonlinearity(self):
        for reg, alpha in REGRESSORS:
            with self.subTest(regressor=reg, alpha=alpha):
                new = new_eval.nonlinearity_testing(*data(), L_MAX, reg, TEST_SIZE, alpha)
                old = legacy_eval.nonlinearity_testing(*data(), L_MAX, reg, TEST_SIZE, alpha)
                assert_same(self, new, old)

    def test_memory(self):
        for reg, alpha in REGRESSORS:
            with self.subTest(regressor=reg, alpha=alpha):
                new = new_eval.memory_testing(*data(), D_MAX, reg, TEST_SIZE, alpha)
                old = legacy_eval.memory_testing(*data(), D_MAX, reg, TEST_SIZE, alpha)
                assert_same(self, new, old)

    def test_nonlinearity_memory_matrix(self):
        for reg, alpha in REGRESSORS:
            with self.subTest(regressor=reg, alpha=alpha):
                new = new_eval.nonlinearity_memory_matrix(*data(), L_MAX, 10, reg, TEST_SIZE, alpha)
                old = legacy_eval.nonlinearity_memory_matrix(*data(), L_MAX, 10, reg, TEST_SIZE, alpha)
                assert_same(self, new, old)

    def test_other_test_sizes(self):
        for test_size in (0.1, 0.5):
            with self.subTest(test_size=test_size):
                new = new_eval.memory_testing(*data(), D_MAX, "Rid", test_size, 1e-2)
                old = legacy_eval.memory_testing(*data(), D_MAX, "Rid", test_size, 1e-2)
                assert_same(self, new, old)

    def test_inputs_are_not_modified(self):
        ip, op = data()
        ip0, op0 = ip.copy(), op.copy()
        new_eval.nonlinearity_testing(ip, op, L_MAX, "Lin", TEST_SIZE, None)
        new_eval.memory_testing(ip, op, D_MAX, "Lin", TEST_SIZE, None)
        np.testing.assert_array_equal(ip, ip0)
        np.testing.assert_array_equal(op, op0)


class TestSavePredictions(unittest.TestCase):
    '''save_predictions_path must not change the scores, and the saved arrays must reproduce them.'''

    def test_nonlinearity_and_memory(self):
        cases = (('nonlinearity', new_eval.nonlinearity_testing, L_MAX, [f'leg{n}' for n in range(1, L_MAX+1)]),
                 ('memory', new_eval.memory_testing, D_MAX, [f'delay{n}' for n in range(D_MAX+1)]))
        with tempfile.TemporaryDirectory() as tmp:
            for name, func, order, labels in cases:
                with self.subTest(name):
                    path = os.path.join(tmp, 'sub', f'{name}.npz')
                    plain = func(*data(), order, "Rid", TEST_SIZE, 1e-2)
                    with quiet():
                        saved = func(*data(), order, "Rid", TEST_SIZE, 1e-2, save_predictions_path=path)
                    assert_same(self, saved, plain)
                    with np.load(path) as d:
                        self.assertEqual(set(d.files), {f'y_test_{l}' for l in labels} | {f'y_test_pred_{l}' for l in labels})
                        for k, label in enumerate(labels):
                            y, pred = d[f'y_test_{label}'], d[f'y_test_pred_{label}']
                            capacity = max(0.0, 1 - np.mean((y - pred)**2) / np.var(y))
                            self.assertAlmostEqual(capacity, plain[1][k], places=12, msg=label)

    def test_ignored_with_cv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'p.npz')
            with quiet():
                res = new_eval.memory_testing(*data(), D_MAX, "Rid", TEST_SIZE, 1e-2, CV=True, type_CV="KFold",
                                              n_splits=N_SPLITS, save_predictions_path=path)
            self.assertFalse(os.path.exists(path))
        assert_same(self, res, legacy_eval.memory_testing(*data(), D_MAX, "Rid", TEST_SIZE, 1e-2, CV=True,
                                                          type_CV="KFold", n_splits=N_SPLITS))


class TestCrossValidationMatchesLegacyEvaluate(unittest.TestCase):
    '''CV moved from utils/evaluate.py into utils/evaluate_cv.py; results must not change.'''

    def test_nonlinearity_cv(self):
        for type_CV in ("KFold", "TimeSeriesSplit"):
            for reg, alpha in REGRESSORS:
                with self.subTest(type_CV=type_CV, regressor=reg, alpha=alpha):
                    new = new_eval.nonlinearity_testing(*data(), L_MAX, reg, TEST_SIZE, alpha,
                                                        CV=True, type_CV=type_CV, n_splits=N_SPLITS)
                    old = legacy_eval.nonlinearity_testing(*data(), L_MAX, reg, TEST_SIZE, alpha,
                                                           CV=True, type_CV=type_CV, n_splits=N_SPLITS)
                    assert_same(self, new, old)

    def test_memory_cv(self):
        for type_CV in ("KFold", "TimeSeriesSplit"):
            for reg, alpha in REGRESSORS:
                with self.subTest(type_CV=type_CV, regressor=reg, alpha=alpha):
                    new = new_eval.memory_testing(*data(), D_MAX, reg, TEST_SIZE, alpha,
                                                  CV=True, type_CV=type_CV, n_splits=N_SPLITS)
                    old = legacy_eval.memory_testing(*data(), D_MAX, reg, TEST_SIZE, alpha,
                                                     CV=True, type_CV=type_CV, n_splits=N_SPLITS)
                    assert_same(self, new, old)

    def test_cv_ignores_test_size(self):
        a = new_eval.memory_testing(*data(), D_MAX, "Rid", 0.25, 1e-2, CV=True, type_CV="KFold", n_splits=N_SPLITS)
        b = new_eval.memory_testing(*data(), D_MAX, "Rid", 0.5, 1e-2, CV=True, type_CV="KFold", n_splits=N_SPLITS)
        assert_same(self, a, b)

    def test_unknown_cv_type_raises(self):
        with self.assertRaises(ValueError):
            new_eval.memory_testing(*data(), D_MAX, "Rid", TEST_SIZE, 1e-2, CV=True, type_CV="LOO", n_splits=N_SPLITS)

    def test_unknown_regressor_raises(self):
        # The old code printed a warning and then crashed with NameError; the new code raises ValueError.
        with self.assertRaises(ValueError):
            new_eval.get_regressor("SVR", 1.0)


class TestMatchesPerStudyLegacyScripts(unittest.TestCase):
    '''The per-study get_*_eval*.py scripts each carry their own scoring functions.'''

    @classmethod
    def setUpClass(cls):
        cls.single = {name: load_module(path, f'legacy_{name}') for name, path in LEGACY_SINGLE_SPLIT_SCRIPTS.items()}
        cls.kfold = load_module(LEGACY_KFOLD, 'legacy_kfold')
        cls.tscv = load_module(LEGACY_TSCV, 'legacy_tscv')
        cls.alpha_cv = load_module(LEGACY_ALPHA_CV, 'legacy_alpha_cv')
        cls.force_sweep = load_module(LEGACY_FORCE_SWEEP, 'legacy_force_sweep')

    def test_single_split_scripts(self):
        for name, mod in self.single.items():
            for reg, alpha in [("Rid", 1e-2), ("Lin", None)]:
                with self.subTest(script=name, regressor=reg):
                    assert_same(self, new_eval.nonlinearity_testing(*data(), L_MAX, reg, TEST_SIZE, alpha),
                                mod.nonlinearity_testing(*data(), L_MAX, reg, TEST_SIZE, alpha), 'nonlinearity')
                    assert_same(self, new_eval.memory_testing(*data(), D_MAX, reg, TEST_SIZE, alpha),
                                mod.memory_testing(*data(), D_MAX, reg, TEST_SIZE, alpha), 'memory')
                    assert_same(self, new_eval.nonlinearity_memory_matrix(*data(), L_MAX, 10, reg, TEST_SIZE, alpha),
                                mod.nonlinearity_memory_matrix(*data(), L_MAX, 10, reg, TEST_SIZE, alpha), 'matrix')

    def test_kfold_script(self):
        # _10fold files: KFold for both nonlinearity and memory
        assert_same(self, new_eval.nonlinearity_testing(*data(), L_MAX, "Rid", TEST_SIZE, 1e-2, CV=True, type_CV="KFold", n_splits=N_SPLITS),
                    self.kfold.nonlinearity_testing(*data(), L_MAX, "Rid", TEST_SIZE, 1e-2, N_SPLITS), 'nonlinearity')
        assert_same(self, new_eval.memory_testing(*data(), D_MAX, "Rid", TEST_SIZE, 1e-2, CV=True, type_CV="KFold", n_splits=N_SPLITS),
                    self.kfold.memory_testing(*data(), D_MAX, "Rid", TEST_SIZE, 1e-2, N_SPLITS), 'memory')

    def test_tscv_script(self):
        # _10tscv files: TimeSeriesSplit for both
        assert_same(self, new_eval.nonlinearity_testing(*data(), L_MAX, "Rid", TEST_SIZE, 1e-2, CV=True, type_CV="TimeSeriesSplit", n_splits=N_SPLITS),
                    self.tscv.nonlinearity_testing(*data(), L_MAX, "Rid", TEST_SIZE, 1e-2, N_SPLITS), 'nonlinearity')
        assert_same(self, new_eval.memory_testing(*data(), D_MAX, "Rid", TEST_SIZE, 1e-2, CV=True, type_CV="TimeSeriesSplit", n_splits=N_SPLITS),
                    self.tscv.memory_testing(*data(), D_MAX, "Rid", TEST_SIZE, 1e-2, N_SPLITS), 'memory')

    def test_alpha_cv_script(self):
        # _10_alphaCV files: KFold for nonlinearity, TimeSeriesSplit for memory
        for alpha in (1e-4, 1e-2, 1e2):
            with self.subTest(alpha=alpha):
                assert_same(self, new_eval.nonlinearity_testing(*data(), L_MAX, "Rid", TEST_SIZE, alpha, CV=True, type_CV="KFold", n_splits=N_SPLITS),
                            self.alpha_cv.nonlinearity_testing(*data(), L_MAX, "Rid", TEST_SIZE, alpha, N_SPLITS), 'nonlinearity')
                assert_same(self, new_eval.memory_testing(*data(), D_MAX, "Rid", TEST_SIZE, alpha, CV=True, type_CV="TimeSeriesSplit", n_splits=N_SPLITS),
                            self.alpha_cv.memory_testing(*data(), D_MAX, "Rid", TEST_SIZE, alpha, N_SPLITS), 'memory')

    def test_force_sweep_script_test_scores(self):
        # The ForceSweep script does not clip train predictions (see CLAUDE.md), so only the test
        # scores (index 1 = capacity test, 3 = R2 test) are expected to match.
        new = new_eval.nonlinearity_testing(*data(), L_MAX, "Rid", TEST_SIZE, 1e-2)
        old = self.force_sweep.nonlinearity_testing(*data(), L_MAX, "Rid", TEST_SIZE, 1e-2)
        assert_same(self, [new[1], new[3]], [old[1], old[3]], 'nonlinearity test')
        new = new_eval.memory_testing(*data(), D_MAX, "Rid", TEST_SIZE, 1e-2)
        old = self.force_sweep.memory_testing(*data(), D_MAX, "Rid", TEST_SIZE, 1e-2)
        assert_same(self, [new[1], new[3]], [old[1], old[3]], 'memory test')

    def test_preprocess_rod_data(self):
        for n in (2, 3, 4):
            history = synthetic_rods_history(num_threads=n, seed=n)
            new = extract_sim_data.preprocess_rod_data(history, n, n)
            self.assertEqual(new.shape, (200, 2*(n*n + 2*n*(n+1))))
            for name, mod in self.single.items():
                with self.subTest(script=name, num_threads=n):
                    np.testing.assert_array_equal(new, mod.preprocess_rod_data(history, n, n))


class TestLoadSimulationData(unittest.TestCase):
    '''utils/extract_sim_data.load_simulation_data vs the legacy loader, on a file in the simulator's format.'''

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.n = 2
        history = synthetic_rods_history(num_threads=cls.n, n_frames=501, duration=2.0)
        time = np.array(history[0]['time'])
        cls.path = os.path.join(cls.tmp.name, 'fake_sim')
        np.savez(cls.path + '.npz', rods_history=history, force_profile=[np.sin(time)], seed_value=1234)
        cls.legacy = load_module(LEGACY_SINGLE_SPLIT_SCRIPTS['ForceSpacing'], 'legacy_loader')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_regenerated_input_matches_legacy_loader(self):
        with quiet():
            new = extract_sim_data.load_simulation_data(self.path, 'npz', 0, self.n, self.n, 1, regenerate_ip=True)
            old = self.legacy.load_simulation_data(self.path, 'npz', 0, self.n, self.n, 1)
        for a, b, what in zip(new, old, ('input', 'output', 'time')):
            with self.subTest(what):
                np.testing.assert_array_equal(a[0], b[0])

    def test_start_and_step(self):
        with quiet():
            full = extract_sim_data.load_simulation_data(self.path, 'npz', 0, self.n, self.n, 1, regenerate_ip=True)
            part = extract_sim_data.load_simulation_data(self.path, 'npz', 10, self.n, self.n, 3, regenerate_ip=True)
        for a, b in zip(full, part):
            np.testing.assert_array_equal(a[0][10::3], b[0])

    def test_missing_file_returns_nan(self):
        with quiet():
            ip, op, t = extract_sim_data.load_simulation_data(os.path.join(self.tmp.name, 'nope'), 'npz', 0, 2, 2, 1)
        self.assertTrue(np.isnan(ip[0]))

    @unittest.expectedFailure
    def test_saved_input_has_same_shape_as_regenerated(self):
        # KNOWN BUG: with regenerate_ip=False the saved force profile (shape (1, T) -> .T -> (T, 1))
        # gets another [:, np.newaxis], so the input is (T, 1, 1) and the regressions fail.
        # fiber_network_evaluations.py --no-regenerate_ip therefore cannot work on spline files.
        with quiet():
            ip, _, _ = extract_sim_data.load_simulation_data(self.path, 'npz', 0, self.n, self.n, 1, regenerate_ip=False)
        self.assertEqual(ip[0].shape, (501, 1))


class TestDriverMatchesLegacyPipeline(unittest.TestCase):
    '''
    fiber_network_evaluations.run() end to end (cached source, outputs redirected to a temp dir)
    vs the loop of fiber_network_evaluations_SAGE.py built on the legacy evaluate.py.
    '''

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # input deliberately not in [-1, 1]: both pipelines rescale it
        self.ip, self.op = synthetic_reservoir(n_samples=2000, n_features=12, seed=3, input_range=(-3.0, 5.0))
        np.savez(os.path.join(self.tmp.name, '0_eval.npz'), input_data=self.ip, output_data=self.op,
                 time_data=np.arange(len(self.ip))/250)

    def run_driver(self, **cfg_kwargs):
        cfg = driver.build_config(dict(case_name="Increasing_Density", idx_list=[0], source="cached",
                                       leg_max_order=L_MAX, max_time_back_seconds=D_MAX/250, n_splits=N_SPLITS, **cfg_kwargs))
        csv_path = os.path.join(self.tmp.name, 'results.csv')
        with mock.patch.object(sr, 'data_folder', lambda case: self.tmp.name), \
             mock.patch.object(sr, 'results_csv_path', lambda case, c: csv_path), \
             mock.patch.object(sr, 'eval_npz_path', lambda case, c, idx, alpha: os.path.join(self.tmp.name, f'{idx}_out_{alpha:g}.npz')), \
             mock.patch.object(sr, 'load_grid', lambda case: np.array([[2, 166.67, 0.0108]])), quiet():
            driver.run(cfg)
        return cfg, pd.read_csv(csv_path)

    def legacy_scores(self, alpha, CV=False, type_nl=None, type_mem=None):
        ip = -1 + (self.ip - np.min(self.ip)) / (np.max(self.ip) - np.min(self.ip)) * (1 - (-1))
        kw_nl = dict(CV=True, type_CV=type_nl, n_splits=N_SPLITS) if CV else {}
        kw_mem = dict(CV=True, type_CV=type_mem, n_splits=N_SPLITS) if CV else {}
        nl = legacy_eval.nonlinearity_testing(ip, self.op.copy(), L_MAX, "Rid", TEST_SIZE, alpha, **kw_nl)
        mem = legacy_eval.memory_testing(ip, self.op.copy(), D_MAX, "Rid", TEST_SIZE, alpha, **kw_mem)
        hm = None if CV else legacy_eval.nonlinearity_memory_matrix(ip, self.op.copy(), L_MAX, D_MAX, "Rid", TEST_SIZE, alpha)
        return nl, mem, hm

    def test_single_split_run(self):
        cfg, df = self.run_driver(alphas=[1e-2], compute_matrix=True)
        nl, mem, hm = self.legacy_scores(1e-2)
        saved = np.load(os.path.join(self.tmp.name, '0_out_0.01.npz'), allow_pickle=True)
        np.testing.assert_array_equal(saved['nonlinearity'], np.asarray(nl))
        np.testing.assert_array_equal(saved['memory'], np.asarray(mem))
        np.testing.assert_array_equal(saved['heatmap'], np.asarray(hm))
        self.assertEqual(json.loads(str(saved['config']))['leg_max_order'], L_MAX)
        row = df.iloc[0]
        # same means fiber_network_evaluations_SAGE.py wrote: sum(list)/len(list)
        self.assertAlmostEqual(row['nonlinearity train'], sum(nl[0])/len(nl[0]), places=12)
        self.assertAlmostEqual(row['nonlinearity test'], sum(nl[1])/len(nl[1]), places=12)
        self.assertAlmostEqual(row['memory train'], sum(mem[0])/len(mem[0]), places=12)
        self.assertAlmostEqual(row['memory test'], sum(mem[1])/len(mem[1]), places=12)
        self.assertAlmostEqual(row['nonlinearity R2 test'], np.mean(nl[3]), places=12)
        self.assertAlmostEqual(row['memory R2 test'], np.mean(mem[3]), places=12)
        self.assertEqual(row['num_threads'], 2)

    def test_alpha_sweep_cv_run(self):
        # replaces fiber_network_evaluations_SAGE_alphaCV.py (KFold nonlinearity, TSS memory)
        alphas = [1e-4, 1e-2, 1.0]
        cfg, df = self.run_driver(alphas=alphas, CV=True, type_CV_nonlin="KFold", type_CV_mem="TimeSeriesSplit",
                                  compute_matrix=False)
        self.assertEqual(sorted(df['alpha']), alphas)
        for alpha in alphas:
            with self.subTest(alpha=alpha):
                nl, mem, _ = self.legacy_scores(alpha, CV=True, type_nl="KFold", type_mem="TimeSeriesSplit")
                row = df[df['alpha'] == alpha].iloc[0]
                self.assertAlmostEqual(row['nonlinearity test'], np.mean(nl[1]), places=12)
                self.assertAlmostEqual(row['memory test'], np.mean(mem[1]), places=12)
                self.assertAlmostEqual(row['nonlinearity train'], np.mean(nl[0]), places=12)
                self.assertAlmostEqual(row['memory train'], np.mean(mem[0]), places=12)

    def test_save_predictions_dir(self):
        pred_dir = os.path.join(self.tmp.name, 'pred')
        cfg, df = self.run_driver(alphas=[1e-2], compute_matrix=False, save_predictions_dir=pred_dir)
        tag = sr.config_tag(cfg, 1e-2)
        self.assertEqual(sorted(os.listdir(pred_dir)), sorted([f'0_predictions_memory__{tag}.npz',
                                                                f'0_predictions_nonlinearity__{tag}.npz']))
        nl, mem, _ = self.legacy_scores(1e-2)   # saving predictions must not change the scores
        self.assertAlmostEqual(df.iloc[0]['nonlinearity test'], np.mean(nl[1]), places=12)
        self.assertAlmostEqual(df.iloc[0]['memory test'], np.mean(mem[1]), places=12)

    def test_rerun_replaces_rows(self):
        self.run_driver(alphas=[1e-2], compute_matrix=False)
        _, df = self.run_driver(alphas=[1e-2], compute_matrix=False)
        self.assertEqual(len(df), 1)

    def test_missing_data_gives_nan_row(self):
        os.remove(os.path.join(self.tmp.name, '0_eval.npz'))
        _, df = self.run_driver(alphas=[1e-2], compute_matrix=False)
        self.assertTrue(np.isnan(df.iloc[0]['nonlinearity test']))


class TestDriverConfig(unittest.TestCase):

    def test_defaults_match_legacy_settings(self):
        cfg = driver.EvalConfig()
        self.assertEqual((cfg.regressor, cfg.alphas, cfg.test_size, cfg.leg_max_order, cfg.max_timesteps_back),
                         ("Rid", [1e-2], 0.25, 10, 250))

    def test_parse_idx(self):
        self.assertEqual(driver.parse_idx("0-3"), [0, 1, 2, 3])
        self.assertEqual(driver.parse_idx("0,4,10-12"), [0, 4, 10, 11, 12])

    def test_precedence_and_unknown_keys(self):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump({'n_splits': 7, 'alphas': [1.0]}, f)
        try:
            cfg = driver.build_config({'n_splits': 3, 'leg_max_order': 4}, {'config': f.name, 'alphas': [5.0]})
        finally:
            os.remove(f.name)
        self.assertEqual((cfg.leg_max_order, cfg.n_splits, cfg.alphas), (4, 7, [5.0]))
        with self.assertRaises(TypeError):
            driver.build_config({'n_split': 3})   # misspelled keys must not be silently ignored

    def test_config_tag(self):
        self.assertEqual(sr.config_tag(driver.EvalConfig()), 'Rid_a1e-02_split0.25_L10_D250')
        cfg = driver.EvalConfig(alphas=[1e-4, 1e-2, 1e2], CV=True)
        self.assertEqual(sr.config_tag(cfg), 'Rid_a1e-04to1e+02x3_cv10-kf-tss_L10_D250')
        self.assertEqual(sr.config_tag(cfg, alpha=1e-2), 'Rid_a1e-02_cv10-kf-tss_L10_D250')

    def test_outputs_never_target_original_files(self):
        for case in sr.STUDIES:
            cfg = driver.EvalConfig(case_name=case)
            self.assertIn('__', os.path.basename(sr.results_csv_path(case, cfg)))
            self.assertNotEqual(os.path.basename(sr.eval_npz_path(case, cfg, 0, 1e-2)), '0_eval.npz')


if __name__ == '__main__':
    unittest.main()
