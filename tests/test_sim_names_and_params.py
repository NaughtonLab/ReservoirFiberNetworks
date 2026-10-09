import os
import unittest

import numpy as np
import pandas as pd

from tests._helpers import load_module, REPO_ROOT

from utils import save_results as sr

'''
The evaluation scripts find each simulation file by rebuilding its name from a grid row. For every
row of every study grid, the new utils/save_results (row_to_params + get_sim_name) must give the
same physical parameters and legacy file name as the legacy fiber_network_evaluations_SAGE.py, and
the same current file name as fiber_simulation.file_name() for a fiber_config_main.py run.
The grid .npz files are tracked in git, so this runs everywhere.
'''

legacy = load_module('fiber_network_evaluations_SAGE.py', 'legacy_sage_driver')

FORCE_SPACING_SPLIT = 112   # Force-Spacing rows >= 112 come from force_spacing_grid_2.npz

def legacy_grid_row(case, idx):
    '''The grid row exactly as the legacy __main__ picked it.'''
    folder = os.path.join(REPO_ROOT, sr.get_study(case)['folder'])
    if case == "GS_Force_Spacing":
        if idx < FORCE_SPACING_SPLIT:
            return np.load(os.path.join(folder, 'force_spacing_grid.npz'), allow_pickle=True)['grid'][idx]
        return np.load(os.path.join(folder, 'force_spacing_grid_2.npz'), allow_pickle=True)['grid'][idx - FORCE_SPACING_SPLIT]
    f, key = sr.get_study(case)['grid_files'][0]
    return np.load(os.path.join(folder, f), allow_pickle=True)[key][idx]

def legacy_name_and_params(case, idx):
    folder = os.path.join(REPO_ROOT, sr.get_study(case)['folder'])
    if case == "Increasing_Density":   # the legacy function reads the module-level `grid`
        legacy.grid = np.load(os.path.join(folder, 'increasing_density_list.npz'), allow_pickle=True)['grid']
    legacy.df = pd.DataFrame()
    name, nh, nv, df = legacy.get_sim_name_and_update_data_frame(legacy_grid_row(case, idx), case, idx, update_df=True)
    return name, nh, nv, df.loc[idx]


class TestSimulationNames(unittest.TestCase):

    def test_grids_cover_all_simulations(self):
        for case, study in sr.STUDIES.items():
            with self.subTest(case=case):
                self.assertGreaterEqual(len(sr.load_grid(case)), study['n_sims'])

    def test_names_and_params_match_legacy(self):
        for case, study in sr.STUDIES.items():
            grid = sr.load_grid(case)
            for idx in range(study['n_sims']):
                with self.subTest(case=case, idx=idx):
                    old_name, nh, nv, old_row = legacy_name_and_params(case, idx)
                    params = sr.row_to_params(case, grid[idx])
                    self.assertEqual(sr.get_sim_name(case, params, idx, fps=250, stepskip=800, legacy=True), old_name)
                    self.assertEqual(params['num_threads'], nh)
                    self.assertEqual(nh, nv)
                    for col, value in old_row.items():
                        if pd.notna(value):
                            self.assertAlmostEqual(params[col], float(value), places=9, msg=col)

    def test_force_spacing_second_grid_offset(self):
        grid = sr.load_grid("GS_Force_Spacing")
        for idx in (FORCE_SPACING_SPLIT - 1, FORCE_SPACING_SPLIT, sr.get_study("GS_Force_Spacing")['n_sims'] - 1):
            with self.subTest(idx=idx):
                np.testing.assert_array_equal(grid[idx], legacy_grid_row("GS_Force_Spacing", idx))

    def test_known_names(self):
        # Spot checks of real file names (Increasing_Density: spacing number in mm + TF block in both formats)
        params = sr.row_to_params("Increasing_Density", sr.load_grid("Increasing_Density")[1])
        for legacy in (False, True):
            self.assertEqual(sr.get_sim_name("Increasing_Density", params, 1, 250, 800, legacy=legacy),
                             '4by4rods_spacing1.0000e+02m_TF1e-02N_PF-5e-02Nspline_fps250_stepskip800_1')
        params = sr.row_to_params("Force_Sweep", sr.load_grid("Force_Sweep")[0])
        self.assertTrue(sr.get_sim_name("Force_Sweep", params, 0, 250, 800, legacy=True).startswith('4by4rods_spacing7.0000e-02m_PF'))
        self.assertTrue(sr.get_sim_name("Force_Sweep", params, 0, 250, 800).startswith('4by4rods_spacing7.0000e-02m_TF1e-02N_PF'))
        # Written by a real fiber_config_main.py run (GS_Tension_Spacing idx 3, see HANDOFF_sim_config_merge.md)
        params = sr.row_to_params("GS_Tension_Spacing", sr.load_grid("GS_Tension_Spacing")[3])
        self.assertEqual(sr.get_sim_name("GS_Tension_Spacing", params, 3, 250, 800),
                         '4by4rods_spacing8.0000e-02m_TF1e-02N_PF-1e-01Nspline_fps250_stepskip800_3')

    def test_names_match_simulator(self):
        # The current name must be what fiber_simulation.file_name() gives for the params fiber_config_main.py builds
        try:
            import fiber_config_main as fcm
        except ImportError as e:   # needs PyElastica
            self.skipTest(f'fiber_config_main not importable: {e}')
        from utils.unit_scaling import unit_scaling
        for case, study in sr.STUDIES.items():
            grid = sr.load_grid(case)
            for idx in range(study['n_sims']):
                with self.subTest(case=case, idx=idx):
                    sim_params = unit_scaling.scale(fcm.build_params(case_name=case, idx=idx), 'mm_g_s')
                    sim = fcm.fiber_simulation(**sim_params)
                    self.assertEqual(sr.get_sim_name(case, sr.row_to_params(case, grid[idx]), idx, sim.rendering_fps, sim.step_skip),
                                     sim.file_name())

    def test_find_sim_file_prefers_current_then_legacy(self):
        import tempfile
        params = sr.row_to_params("GS_Force_Spacing", sr.load_grid("GS_Force_Spacing")[5])
        current = sr.get_sim_name("GS_Force_Spacing", params, 5, 250, 800)
        old = sr.get_sim_name("GS_Force_Spacing", params, 5, 250, 800, legacy=True)
        with tempfile.TemporaryDirectory() as folder:
            find = lambda: sr.find_sim_file("GS_Force_Spacing", params, 5, 250, 800, folder=folder)
            self.assertEqual(find(), os.path.join(folder, current))          # nothing on disk -> current name
            open(os.path.join(folder, old + '.npz'), 'w').close()
            self.assertEqual(find(), os.path.join(folder, old))              # only the legacy file
            open(os.path.join(folder, current + '.npz'), 'w').close()
            self.assertEqual(find(), os.path.join(folder, current))          # both -> current

    def test_thread_length_rule(self):
        for case in sr.STUDIES:
            params = sr.row_to_params(case, sr.load_grid(case)[0])
            with self.subTest(case=case):
                if case == "Increasing_Density":
                    self.assertEqual(params['length(mm)'], 500.0)
                else:
                    self.assertAlmostEqual(params['length(mm)'], params['spacing(mm)']*(params['num_threads']+1))


if __name__ == '__main__':
    unittest.main()
