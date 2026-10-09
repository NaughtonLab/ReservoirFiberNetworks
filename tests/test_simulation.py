import os
import tempfile
import unittest

import numpy as np
from scipy.interpolate import CubicSpline

from tests._helpers import quiet
from tests.sim_case import run_case, case_params, positions, package_versions, GOLDEN_PATH

try:
    import elastica   # noqa: F401
    HAVE_ELASTICA = True
except ImportError:
    HAVE_ELASTICA = False

'''
Does the simulator still behave the same after upgrading PyElastica / numba / numpy?

1. A short 2x2 simulation (tests/sim_case.py) is run through fiber_simulation.launch_sim and
   compared with a golden reference saved from a trusted environment
   (tests/reference/sim_golden_2x2.npz, written by `python tests/sim_case.py --write-golden`).
2. Physical invariants that do not depend on any reference: boundary conditions, joints holding
   the crossings together, planar motion, x<->y symmetry without point force, the point force
   input and its regeneration in the evaluation code.
3. Canaries for the PyElastica behaviour this code relies on, so an upgrade fails with a clear
   message instead of silently changing results.

Baseline: the golden file was written with the committed simulator (HEAD 668d016) in the trusted
env fiber_network_venv_conda: PyElastica 0.3.1.post1 with a locally patched elastica/_rotations.py
(its sha1 is stored in the golden 'versions'), numba 0.55.2, numpy 1.22.4. The current code
reproduces it bit for bit there.
Measured against that baseline: PyElastica 0.3.3.post2 differs by 1.45e-4 of the displacement, so
test_matches_golden_reference FAILS on 0.3.3 / 1.0.0 by design - an upgrade changes the physics
relative to the published results by that much. For scale, 0.3.3 vs 1.0.0 (+ numba 0.68, numpy
2.2) differ by only ~2e-10, and a 0.01 % change of the damping constant gives 3.4e-6.
'''

GOLDEN_RTOL = 1e-6   # max |position - golden| / max |displacement|
JOINT_GAP_TOL = 1e-4  # mm; FixedJoint with k = 1e9 keeps the crossings within ~1e-6 mm

P = case_params()
PF_SCALED = P['point_force_mag'] * 1e6   # N -> g mm / s^2 (mm_g_s)
N_FRAMES = int(round(P['duration'] * P['rendering_fps'])) + 1


@unittest.skipUnless(HAVE_ELASTICA, 'PyElastica is not installed')
class TestSimulationRun(unittest.TestCase):
    '''One run of the test case shared by all tests of this class (~15-30 s).'''

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sim, cls.path = run_case(cls.tmp.name)
        if cls.path is not None:
            with np.load(cls.path, allow_pickle=True) as d:
                cls.rods_history = list(d['rods_history'])
                cls.force_profile = d['force_profile']
                cls.seed_value = int(d['seed_value'])
                cls.files = d.files
            cls.pos = positions(cls.rods_history)        # (rod, frame, xyz, node)
            cls.disp = cls.pos - cls.pos[:, :1]
            cls.time = np.array(cls.rods_history[0]['time'])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        if self.path is None:
            self.fail('The simulation stopped at NaN, nothing was saved '
                      f'(versions: {package_versions()}). See test_joint_rotation_canary.')

    def test_saves_expected_file(self):
        self.assertEqual(os.path.basename(self.path),
                         '2by2rods_spacing1.0000e+02m_TF1e-02N_PF-5e-03Nspline_fps250_stepskip800_0.npz')
        self.assertEqual(set(self.files), {'rods_history', 'force_profile', 'seed_value'})
        self.assertEqual(len(self.rods_history), 4)
        self.assertEqual(self.seed_value, 1234)

    def test_sampling(self):
        self.assertEqual(self.pos.shape, (4, N_FRAMES, 3, 31))
        np.testing.assert_allclose(np.diff(self.time), 1/P['rendering_fps'], rtol=1e-6)
        self.assertTrue(np.all(np.isfinite(self.pos)))

    def test_matches_golden_reference(self):
        if not os.path.exists(GOLDEN_PATH):
            self.skipTest(f'no golden file; create it with python tests/sim_case.py --write-golden')
        with np.load(GOLDEN_PATH) as g:
            golden, golden_force, golden_versions = g['positions'], g['force_profile'], str(g['versions'])
        self.assertEqual(self.pos.shape, golden.shape)
        scale = np.abs(golden - golden[:, :1]).max()
        err = np.abs(self.pos - golden).max() / scale
        self.assertLessEqual(err, GOLDEN_RTOL,
                             f'trajectory differs from the golden run by {err:.2e} of the displacement '
                             f'(golden: {golden_versions}, now: {package_versions()})')
        np.testing.assert_allclose(self.force_profile, golden_force, rtol=1e-12, atol=0)

    def test_boundary_conditions(self):
        # first node: fully fixed. last node: free only along the rod (horizontal: x, vertical: y)
        np.testing.assert_array_equal(self.disp[:, :, :, 0], 0.0)
        for rod in range(4):
            along = 0 if rod < 2 else 1
            for axis in (0, 1, 2):
                if axis != along:
                    np.testing.assert_array_equal(self.disp[rod, :, axis, -1], 0.0, err_msg=f'rod {rod}, axis {axis}')
        # tension pulls the free end outwards
        self.assertTrue(np.all(self.disp[:2, -1, 0, -1] > 0))
        self.assertTrue(np.all(self.disp[2:, -1, 1, -1] > 0))

    def test_joints_hold_crossings_together(self):
        h_idx, v_idx = self.sim.hor_connect_idx.astype(int), self.sim.vert_connect_idx.astype(int)
        np.testing.assert_array_equal(h_idx, [10, 20])
        for j in range(2):
            for i in range(2):
                gap = np.abs(self.pos[j][:, :, h_idx[i]] - self.pos[2+i][:, :, v_idx[j]]).max()
                self.assertLess(gap, JOINT_GAP_TOL, f'horizontal {j} / vertical {i}')

    def test_motion_is_planar(self):
        np.testing.assert_array_equal(self.pos[:, :, 2, :], 0.0)

    def test_point_force_drives_stimulated_rod(self):
        # 2x2 network: the force acts on horizontal rod 0 at node n_elem//2 = 15 (spread +-2)
        y = np.abs(self.disp[:, :, 1, :]).max(axis=1)     # (rod, node)
        self.assertEqual(int(np.argmax(y[0])), 15)
        self.assertGreater(y[0].max(), 0.1)                 # mm
        self.assertGreater(y[0].max(), 10*y[1:].max())

    def test_force_profile_is_seeded_spline(self):
        # independent re-implementation of the input in fiber_simulation.launch_sim
        np.random.seed(1234)
        sample_time = int(np.ceil(P['duration']))
        x = np.linspace(0, sample_time, sample_time*P['sample_freq'] + 1)
        y = np.random.uniform(-1, 1, size=sample_time*P['sample_freq'] + 1)
        y[0] = 0.0
        expected = CubicSpline(x, y)(self.time) * PF_SCALED
        np.testing.assert_allclose(self.force_profile[0], expected, rtol=1e-12, atol=0)

    def test_evaluation_regenerates_the_same_input(self):
        from utils.extract_sim_data import load_simulation_data
        with quiet():
            ip, op, t = load_simulation_data(self.path[:-4], 'npz', 0, 2, 2, 1, regenerate_ip=True)
        np.testing.assert_allclose(ip[0][:, 0], self.force_profile[0] / PF_SCALED, rtol=0, atol=1e-12)
        self.assertEqual(op[0].shape, (N_FRAMES, 32))
        self.assertTrue(np.all(np.isfinite(op[0])))


@unittest.skipUnless(HAVE_ELASTICA, 'PyElastica is not installed')
class TestSimulationWithoutPointForce(unittest.TestCase):

    def test_tension_only_is_symmetric(self):
        # With no point force the network is symmetric under swapping x and y: horizontal rod j
        # must deform exactly like vertical rod j with the axes swapped.
        with tempfile.TemporaryDirectory() as tmp:
            _, path = run_case(tmp, point_force_mag=0.0, duration=0.1)
            self.assertIsNotNone(path, 'stopped at NaN')
            with np.load(path, allow_pickle=True) as d:
                pos = positions(d['rods_history'])
        disp = pos - pos[:, :1]
        scale = np.abs(disp).max()
        self.assertGreater(scale, 1e-4)   # the tension does stretch the rods
        for j in range(2):
            np.testing.assert_allclose(disp[j, :, 0], disp[2+j, :, 1], rtol=0, atol=1e-9*scale)
            np.testing.assert_allclose(disp[j, :, 1], disp[2+j, :, 0], rtol=0, atol=1e-9*scale)


@unittest.skipUnless(HAVE_ELASTICA, 'PyElastica is not installed')
class TestElasticaCompatibility(unittest.TestCase):

    def test_stepper_interface(self):
        # fiber_simulation.launch_sim steps with extend_stepper_interface(...) + do_step(...).
        # PyElastica 1.0.0 deprecates do_step and its deprecation path crashes
        # (ModuleNotFoundError: No module named 'warning'); there the loop must call
        # self.StatefulStepper.step(self.simulator, current_time, self.sim_dt) instead,
        # which also exists in 0.3.3 (not in 0.3.1).
        from elastica import CosseratRod, PositionVerlet
        from elastica.timestepper import extend_stepper_interface
        from fiber_simulation_main import BaseSimulator
        sim = BaseSimulator()
        sim.append(CosseratRod.straight_rod(n_elements=5, start=np.zeros(3), direction=np.array([1.0, 0, 0]),
                                            normal=np.array([0, 1.0, 0]), base_length=1.0, base_radius=0.01,
                                            density=1.0, youngs_modulus=1e3))
        sim.finalize()
        stepper = PositionVerlet()
        do_step, stages_and_updates = extend_stepper_interface(stepper, sim)
        with quiet():
            t = do_step(stepper, stages_and_updates, sim, 0.0, 1e-4)
        self.assertAlmostEqual(float(t), 1e-4)

    def test_joint_rotation_canary(self):
        # fiber_simulation.add_threads passes rest_rotation_matrix = C_h @ C_v to FixedJoint, while
        # FixedJoint compares it with C_h @ C_v.T. Every joint therefore starts 180 deg away from its
        # rest orientation, where the rotation log-map is singular, and the in-plane motion keeps it
        # there. PyElastica 0.3.3 / 1.0.0, and the trusted 0.3.1.post1 env with its patched
        # _rotations.py, return a zero rotation vector at 180 deg, so the joints apply no torque
        # (kt has no effect). An unpatched 0.3.1.post1 (pip install from requirements.txt) returns
        # NaN and the simulation stops at the second step. If this test fails, the joint torques -
        # and therefore all results - differ from the published ones.
        from elastica import CosseratRod
        from elastica._rotations import _inv_rotate
        rods = [CosseratRod.straight_rod(n_elements=4, start=np.zeros(3), direction=d, normal=n, base_length=1.0,
                                         base_radius=0.01, density=1.0, youngs_modulus=1.0)
                for d, n in ((np.array([1.0, 0, 0]), np.array([0, 1.0, 0])), (np.array([0, 1.0, 0]), np.array([1.0, 0, 0])))]
        C_h, C_v = rods[0].director_collection[..., 2], rods[1].director_collection[..., 2]
        dev_rot = (C_h @ C_v.T).T @ (C_h @ C_v)   # as computed inside FixedJoint.apply_torques
        self.assertAlmostEqual(np.trace(dev_rot), -1.0)   # 180 deg
        rot_vec = _inv_rotate(np.dstack([np.eye(3), dev_rot.T])).squeeze()
        self.assertTrue(np.all(np.isfinite(rot_vec)), f'rotation vector at 180 deg is {rot_vec}')
        np.testing.assert_array_equal(rot_vec, 0.0)


class TestUnitScaling(unittest.TestCase):

    def test_mm_g_s(self):
        from utils.unit_scaling import unit_scaling
        p = unit_scaling.scale(case_params(), "mm_g_s")
        self.assertAlmostEqual(p['thread_length'], 300.0)
        self.assertAlmostEqual(p['thread_diameter'], 2.0)
        self.assertAlmostEqual(p['dx'], 10.0)
        self.assertAlmostEqual(p['youngs_modulus'], 100e6)
        self.assertAlmostEqual(p['density'], 1e-3)
        self.assertAlmostEqual(p['tension_force'], 1e4)
        self.assertAlmostEqual(p['point_force_mag'], -5e3)
        self.assertAlmostEqual(p['duration'], 0.5)

    @unittest.expectedFailure
    def test_mm_mg_ms(self):
        # KNOWN BUG: mm_mg_ms scales sample_freq to 0.005 knots per ms, a float, so the spline input
        # cannot be built (TypeError in np.linspace; here 500 ms * 0.005 = 2.5 knots is not even an
        # integer). The same input must come out as with mm_g_s, only with time in ms.
        from utils.unit_scaling import unit_scaling
        from utils.forces.spline_input import generate_spline_inputs
        p = unit_scaling.scale(case_params(), "mm_mg_ms")
        self.assertAlmostEqual(p['duration'], 500.0)
        t = np.linspace(0, P['duration'], 11)
        reference = generate_spline_inputs(P['duration'], P['sample_freq'], 1234)[0](t)
        scaled = generate_spline_inputs(p['duration'], p['sample_freq'], 1234)[0](t*1e3)
        np.testing.assert_allclose(scaled, reference, atol=1e-12)


class TestInputRegeneration(unittest.TestCase):

    def test_numpy_legacy_random_stream(self):
        # The spline input (and its regeneration at evaluation time) relies on np.random.seed(1234)
        # + np.random.uniform giving the same numbers in every numpy version.
        np.random.seed(1234)
        np.testing.assert_allclose(np.random.uniform(-1, 1, size=6),
                                   [-0.6169611, 0.24421754, -0.12454452, 0.57071717, 0.55995162, -0.45481479],
                                   rtol=0, atol=5e-9)

    def test_regeneration_with_integer_duration(self):
        # Regression test: the saved time stamps accumulate rounding error, e.g. a 1 s run ends at
        # 1.0000000000044 and fiber_config_main.py (10 s) at 10.000000000374. The old
        # load_simulation_data used ceil(time[-1]) and rebuilt one second too many (10 s run:
        # nonlinearity test 0.293 instead of 0.321). It now uses rint(time[-1]) (or an explicit
        # duration), which is right for the integer durations of every study.
        from utils.extract_sim_data import load_simulation_data
        from tests._helpers import synthetic_rods_history
        history = synthetic_rods_history(num_threads=2, n_frames=251, duration=1.0)
        time = np.array(history[0]['time'])
        time[-1] = 1.0000000000044342
        for rod in history:
            rod['time'] = list(time)
        np.random.seed(1234)
        y = np.random.uniform(-1, 1, size=6)
        y[0] = 0.0
        force = CubicSpline(np.linspace(0, 1, 6), y)(time) * PF_SCALED
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'sim')
            np.savez(path + '.npz', rods_history=history, force_profile=[force], seed_value=1234)
            with quiet():
                ip, _, _ = load_simulation_data(path, 'npz', 0, 2, 2, 1, regenerate_ip=True)
        np.testing.assert_allclose(ip[0][:, 0], force / PF_SCALED, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
