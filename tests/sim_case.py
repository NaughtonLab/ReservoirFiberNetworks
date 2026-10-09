import os
import sys
import glob
import argparse
import importlib.metadata

import numpy as np

if __package__ in (None, ''):   # run as a script: python tests/sim_case.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests._helpers import REFERENCE_DIR, quiet

'''
The small simulation used by tests/test_simulation.py, and the script that writes its golden
reference (tests/reference/sim_golden_2x2.npz).

Same physics settings as fiber_config_main.py (2x2 network, E = 100 MPa, spline point force with
spread, dt = 5e-6 s), only a shorter thread (300 mm) and 0.5 s of simulated time so it runs in
~15-30 s.

Regenerate the golden file ONLY from an environment whose results you trust, e.g. before
upgrading packages:
    python tests/sim_case.py --write-golden
'''

GOLDEN_PATH = os.path.join(REFERENCE_DIR, 'sim_golden_2x2.npz')
PACKAGES = ['pyelastica', 'numba', 'numpy', 'scipy']

def case_params():
    '''Parameters in SI units, as in the fiber_config_*.py scripts.'''
    return {
        'num_horizontal_threads': 2,
        'num_vertical_threads': 2,
        'network_origin': np.zeros((3,)),
        'thread_length': 300e-3,
        'thread_diameter': 2e-3,
        'dx': 10e-3,
        'youngs_modulus': 100e6,
        'density': 1e3,
        'tension_force': 1e-2,
        'point_force_mag': -5e-3,
        'SPREAD_PF': True,
        'TYPE_PF': "spline",
        'sample_freq': 5,
        'damping_constant': 10,
        'filter_order': 6,
        'k': 1e9,
        'kt': 1e9,
        'nu': 0.0,
        'duration': 0.5,
        'sim_dt': 5e-6,
        'rendering_fps': 250,
        'STOP_AT_NAN': True,
        'SAVE': True,
        'VIDEO': False,
        'scaling_type': "mm_g_s",
        'file_type': 'npz',
    }

def run_case(out_dir, **overrides):
    '''Runs the case through fiber_simulation.launch_sim. Returns (sim, saved npz path or None if it stopped at NaN).'''
    from fiber_simulation_main import fiber_simulation
    from utils.unit_scaling import unit_scaling
    params = case_params()
    params.update(overrides)
    params = unit_scaling.scale(params, params['scaling_type'])
    params['loc'] = os.path.join(out_dir, '')
    sim = fiber_simulation(**params)
    with quiet():
        sim.launch_sim()
    saved = glob.glob(os.path.join(out_dir, '*.npz'))
    return sim, (saved[0] if saved else None)

def positions(rods_history):
    '''(n_rods, n_frames, 3, n_nodes)'''
    return np.stack([np.array(r['position']) for r in rods_history])

def package_versions():
    '''Package versions, plus a hash of elastica/_rotations.py: the trusted 0.3.1.post1 env has a locally
    patched _rotations.py (no NaN at a 180 deg joint), which the version number alone does not show.'''
    import hashlib
    import elastica._rotations
    versions = {p: importlib.metadata.version(p) for p in PACKAGES}
    with open(elastica._rotations.__file__, 'rb') as f:
        text = f.read().replace(b'\r\n', b'\n')
    versions['elastica/_rotations.py sha1'] = hashlib.sha1(text).hexdigest()[:12]
    return versions

def write_golden():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _, path = run_case(tmp)
        if path is None:
            raise RuntimeError("The simulation stopped at NaN in this environment; not writing a golden file.")
        with np.load(path, allow_pickle=True) as d:
            rods_history = d['rods_history']
            np.savez_compressed(GOLDEN_PATH,
                                positions=positions(rods_history),
                                time=np.array(rods_history[0]['time']),
                                force_profile=d['force_profile'],
                                file_name=os.path.basename(path),
                                versions=repr(package_versions()))
    print(f"Wrote {GOLDEN_PATH} with {package_versions()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Writes the golden reference of the test simulation.")
    parser.add_argument('--write-golden', action='store_true', help=f'overwrite {GOLDEN_PATH}')
    if parser.parse_args().write_golden:
        write_golden()
    else:
        parser.print_help()
