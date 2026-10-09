import os
import sys
import types
import runpy
import unittest
import contextlib

import numpy as np

from tests._helpers import load_module, quiet, REPO_ROOT, REFERENCE_DIR

from utils.unit_scaling import unit_scaling

'''
fiber_config_main.py replaced the per-study fiber_config_<study>.py scripts. For every row of every
study grid, every Output_Reduction run of the Slurm scripts, and the default configuration, the
parameters it hands to fiber_simulation (after unit scaling) must be identical to those of the
legacy script. Only 'loc' differs, because the legacy scripts point to folders that no longer exist.

The legacy scripts are run as __main__ with a fake fiber_simulation that records its kwargs, so
nothing is simulated and PyElastica is not needed. Their wrong grid paths are redirected to the
real files. The default configuration is compared with the frozen copy in
tests/reference/fiber_config_main_legacy.py. A legacy study script that has been deleted is
skipped; one moved to legacy/ is still found.
'''

# legacy script, case name, index argument, number of grid rows (None: Output_Reduction)
LEGACY_SCRIPTS = [
    ('fiber_config_GSForceSpacing.py', 'GS_Force_Spacing', '--grid_idx', 252),
    ('fiber_config_SweepForceCliff.py', 'Force_Sweep', '--list_idx', 9),
    ('fiber_config_GSTensionSpacing.py', 'GS_Tension_Spacing', '--grid_idx', 42),
    ('fiber_config_GSThreadSpacing_NL.py', 'GS_Thread_Spacing_NL', '--grid_idx', 84),
    ('fiber_config_GSThreadSpacing_MC.py', 'GS_Thread_Spacing_MC', '--grid_idx', 84),
    ('fiber_config_IncreasingDensity.py', 'Increasing_Density', '--grid_idx', 6),
]
# (num_threads, --point_force_mag in tenths of N) as in Simulations/SAGE/OutputReduction/*.sh
OUTPUT_REDUCTION_RUNS = [(2, 1), (4, 1), (6, 2), (10, 10)]

# Folders the legacy scripts use -> the files that exist
LEGACY_PATHS = {
    'Simulations/SAGE/GridSearch/Tension_Spacing': 'Simulations/SAGE/GridSearch/TensionSpacing',
    'Simulations/SAGE/ThreadSpacing': 'Simulations/SAGE/GridSearch/ThreadSpacing',
    'Simulations/SAGE/ForceCliff': 'Simulations/SAGE/GridSearch/ForceSweep',
    '/projects/naughton/Apoorva/fiber_network_project/fiber_network/Simulations/SAGE/IncDensity/density_list.npz':
        'Simulations/SAGE/IncreasingDensity/increasing_density_list.npz',
}

LEGACY_DEFAULT = os.path.join(REFERENCE_DIR, 'fiber_config_main_legacy.py')

def find_legacy(script):
    for folder in ('', 'legacy'):
        path = os.path.join(REPO_ROOT, folder, script)
        if os.path.exists(path):
            return path
    return None

class Recorder:
    '''Stand-in for fiber_simulation: keeps the kwargs of every construction.'''
    calls = []
    def __init__(self, **kwargs):
        Recorder.calls.append(kwargs)
    def launch_sim(self):
        pass

@contextlib.contextmanager
def fake_simulator():
    '''Replaces fiber_simulation_main and redirects the legacy grid paths, then restores both.'''
    fake = types.ModuleType('fiber_simulation_main')
    fake.fiber_simulation = Recorder
    saved_module = sys.modules.get('fiber_simulation_main')
    np_load = np.load

    def redirected_load(path, *args, **kwargs):
        path = str(path)
        for old, new in LEGACY_PATHS.items():
            if path.startswith(old):
                path = new + path[len(old):]
        return np_load(path, *args, **kwargs)

    cwd = os.getcwd()
    sys.modules['fiber_simulation_main'] = fake
    np.load = redirected_load
    os.chdir(REPO_ROOT)   # the legacy scripts use paths relative to the repo root
    try:
        yield
    finally:
        os.chdir(cwd)
        np.load = np_load
        if saved_module is None:
            sys.modules.pop('fiber_simulation_main', None)
        else:
            sys.modules['fiber_simulation_main'] = saved_module

def run_legacy(path, argv):
    '''Scaled params of one legacy run.'''
    Recorder.calls = []
    saved_argv = sys.argv
    sys.argv = [path] + argv
    try:
        with quiet():
            runpy.run_path(path, run_name='__main__')
    finally:
        sys.argv = saved_argv
    assert len(Recorder.calls) == 1, f'{path} built {len(Recorder.calls)} simulations'
    return Recorder.calls[0]

def scaled(params):
    return unit_scaling.scale(params=params, scaling_type=params['scaling_type'])


class TestSimConfigParity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with fake_simulator():
            cls.fcm = load_module('fiber_config_main.py', 'fiber_config_main_under_test')

    def build(self, **kwargs):
        with fake_simulator():
            return scaled(self.fcm.build_params(**kwargs))

    def assertSameParams(self, legacy, new):
        legacy = {k: v for k, v in legacy.items() if k != 'loc'}
        new = {k: v for k, v in new.items() if k != 'loc'}
        self.assertEqual(sorted(legacy), sorted(new), 'parameter keys differ')
        for key, value in legacy.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(new[key], value, err_msg=key)
            else:
                # exact: the configs must not change a single bit of any parameter
                self.assertEqual(new[key], value, key)
                self.assertEqual(isinstance(new[key], bool), isinstance(value, bool), key)
                self.assertEqual(isinstance(new[key], str), isinstance(value, str), key)

    def test_study_grids_match_legacy_scripts(self):
        for script, case, flag, n_rows in LEGACY_SCRIPTS:
            path = find_legacy(script)
            with self.subTest(case=case):
                if path is None:
                    self.skipTest(f'{script} not found')
                self.assertEqual(len(self.fcm.load_grid(case)), n_rows)
            if path is None:
                continue
            for idx in range(n_rows):
                with self.subTest(case=case, idx=idx):
                    with fake_simulator():
                        legacy = run_legacy(path, [flag, str(idx)])
                    self.assertSameParams(legacy, self.build(case_name=case, idx=idx))

    def test_output_reduction_matches_legacy_script(self):
        path = find_legacy('fiber_config_OutputReduction.py')
        if path is None:
            self.skipTest('fiber_config_OutputReduction.py not found')
        for num_threads, force_tenths in OUTPUT_REDUCTION_RUNS:
            with self.subTest(num_threads=num_threads, force_tenths=force_tenths):
                with fake_simulator():
                    legacy = run_legacy(path, ['--num_threads', str(num_threads), '--point_force_mag', str(force_tenths)])
                new = self.build(case_name='Output_Reduction', num_threads=num_threads, force=force_tenths/10)
                self.assertSameParams(legacy, new)

    def test_default_matches_frozen_config(self):
        with fake_simulator():
            legacy = run_legacy(LEGACY_DEFAULT, [])
        new = self.build()
        self.assertSameParams(legacy, new)
        self.assertEqual(new['loc'], legacy['loc'])

    def test_output_folders(self):
        # loc is the study's data folder; the study folder itself is tracked in git
        for case in self.fcm.SIM_CASES:
            kwargs = dict(num_threads=4, force=0.1) if case == 'Output_Reduction' else dict(idx=0)
            loc = self.build(case_name=case, **kwargs)['loc']
            with self.subTest(case=case):
                data_subdir = self.fcm.case_setting(case, 'data_subdir')
                self.assertEqual(os.path.basename(os.path.normpath(loc)), data_subdir)
                self.assertTrue(os.path.isdir(os.path.dirname(os.path.normpath(loc))), loc)
        self.assertEqual(self.build(case_name='GS_Force_Spacing', idx=0, output_dir='somewhere')['loc'],
                         os.path.join('somewhere', ''))

    def test_force_spacing_second_grid(self):
        # rows >= 112 come from force_spacing_grid_2.npz
        folder = os.path.join(REPO_ROOT, 'Simulations/SAGE/GridSearch/ForceSpacing')
        grid_2 = np.load(os.path.join(folder, 'force_spacing_grid_2.npz'))['grid']
        params = self.build(case_name='GS_Force_Spacing', idx=112)
        self.assertEqual(params['point_force_mag'], -grid_2[0][0] * 1e6)
        self.assertEqual(params['n_file'], 112)

    def test_grid_file_override(self):
        # the older tension grid starts at 1e-5 N instead of 0.01 N
        folder = os.path.join(REPO_ROOT, 'Simulations/SAGE/GridSearch/TensionSpacing')
        default = np.load(os.path.join(folder, 'tension_spacing_force_grid_new.npz'))['grid']
        other = np.load(os.path.join(folder, 'tension_spacing_force_grid.npz'))['grid']
        self.assertEqual(self.build(case_name='GS_Tension_Spacing', idx=0)['tension_force'], default[0][0] * 1e6)
        # bare file name -> study folder; a path is used as given; grid_key falls back to the only array
        for grid_file, grid_key in (('tension_spacing_force_grid.npz', None),
                                    (os.path.join(folder, 'tension_spacing_force_grid.npz'), None),
                                    ('tension_spacing_force_grid.npz', 'not_a_key')):
            with self.subTest(grid_file=grid_file, grid_key=grid_key):
                params = self.build(case_name='GS_Tension_Spacing', idx=0, grid_files=[grid_file], grid_key=grid_key)
                self.assertEqual(params['tension_force'], other[0][0] * 1e6)
        # Force_Sweep reads its 'sweep' key from the study default
        sweep = np.load(os.path.join(REPO_ROOT, 'Simulations/SAGE/GridSearch/ForceSweep/forces_sweep.npz'))['sweep']
        self.assertEqual(self.build(case_name='Force_Sweep', idx=2)['point_force_mag'], -sweep[2] * 1e6)

    def test_idx_out_of_range(self):
        with self.assertRaises(IndexError):
            self.build(case_name='Increasing_Density', idx=6)

    def test_cli(self):
        def parse(*argv):
            saved = sys.argv
            sys.argv = ['fiber_config_main.py', *argv]
            try:
                with quiet(), contextlib.redirect_stderr(open(os.devnull, 'w')):
                    return self.fcm.parse_args()
            finally:
                sys.argv = saved

        self.assertIsNone(parse().case_name)
        for flag in ('--idx', '--grid_idx', '--list_idx'):
            with self.subTest(alias=flag):
                args = parse('--case_name', 'Force_Sweep', flag, '3')
                self.assertEqual((args.case_name, args.idx), ('Force_Sweep', 3))
        self.assertEqual(parse('--case', 'GS_Force_Spacing', '--idx', '1').case_name, 'GS_Force_Spacing')

        invalid = [
            ('--case_name', 'GS_Force_Spacing'),                         # grid study without --idx
            ('--case_name', 'Output_Reduction', '--num_threads', '4'),   # missing --force
            ('--case_name', 'Output_Reduction', '--num_threads', '4', '--force', '0.1', '--idx', '0'),
            ('--idx', '3'),                                             # default case has no grid
            ('--force', '1'),
            ('--case_name', 'GS_Force_Spacing', '--idx', '0', '--num_threads', '4'),
            ('--case_name', 'Not_A_Study', '--idx', '0'),
        ]
        for argv in invalid:
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    parse(*argv)


if __name__ == '__main__':
    unittest.main()
