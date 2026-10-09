import os
import io
import sys
import importlib.util
import contextlib

import numpy as np

'''
Shared helpers for the test suite. Tests are plain unittest, so no extra package is needed:

    python -m unittest discover -s tests -t . -v            (from the repo root)

Set FIBER_SLOW_TESTS=1 to also run the slow checks (heatmaps, full alpha sweep, every
Increasing_Density simulation instead of only idx 0).
'''

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_ROOT, 'tests')
REFERENCE_DIR = os.path.join(TESTS_DIR, 'reference')

# The code imports `utils...` and `fiber_simulation_main` relative to the repo root.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')   # no plot windows from the imported scripts

SLOW = os.environ.get('FIBER_SLOW_TESTS', '0') == '1'

INC_DENSITY_FOLDER = os.path.join(REPO_ROOT, 'Simulations', 'SAGE', 'IncreasingDensity')
INC_DENSITY_DATA = os.path.join(INC_DENSITY_FOLDER, 'Data')

def load_module(rel_path, name):
    '''Imports a script by file path (the legacy scripts are not packages). Their code is under
    `if __name__ == "__main__"`, so importing only defines their functions.'''
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@contextlib.contextmanager
def quiet():
    '''Silences the print() calls of the simulation and evaluation code.'''
    with contextlib.redirect_stdout(io.StringIO()):
        yield

def synthetic_reservoir(n_samples=3000, n_features=16, seed=0, input_range=(-1.0, 1.0)):
    '''
    Small stand-in for a simulated reservoir: a smooth random input u(t) and features that are
    nonlinear functions of delayed copies of u plus noise, so both nonlinearity and memory
    capacities are non-trivial and some predictions exceed |1| (exercises the clipping branch).
    Returns input (n, 1) and output (n, n_features).
    '''
    rng = np.random.default_rng(seed)
    knots = rng.uniform(-1, 1, size=n_samples//20 + 2)
    u = np.interp(np.linspace(0, len(knots)-1, n_samples), np.arange(len(knots)), knots)
    features = []
    for k in range(n_features):
        delay = int(rng.integers(0, 25))
        a, b, c = rng.normal(size=3)
        ud = np.roll(u, delay)
        ud[:delay] = 0.0
        features.append(np.tanh(a*ud + b*ud**2 + c*ud**3) + 0.05*rng.normal(size=n_samples))
    output = np.column_stack(features)
    output = (output - output.mean(axis=0)) / output.std(axis=0)
    lo, hi = input_range
    input = (lo + (u - u.min()) / (u.max() - u.min()) * (hi - lo))[:, np.newaxis]
    return input, output

def synthetic_rods_history(num_threads, n_elem=30, n_frames=200, seed=0, duration=2.0):
    '''
    rods_history in the format the simulator saves (list of dicts with 'time' and 'position'
    of shape (n_frames, 3, n_elem+1)), for 2*num_threads straight rods with random wiggles.
    '''
    rng = np.random.default_rng(seed)
    time = np.linspace(0, duration, n_frames)
    s = np.linspace(-150.0, 150.0, n_elem+1)
    history = []
    for r in range(2*num_threads):
        offset = -150.0 + 300.0*(r % num_threads + 1)/(num_threads + 1)
        base = np.zeros((3, n_elem+1))
        if r < num_threads:   # horizontal: along x at y = offset
            base[0], base[1] = s, offset
        else:                 # vertical: along y at x = offset
            base[0], base[1] = offset, s
        wiggle = rng.normal(scale=0.5, size=(n_frames, 2, n_elem+1)).cumsum(axis=0) * 0.01
        pos = np.repeat(base[np.newaxis], n_frames, axis=0)
        pos[:, 0:2, :] += wiggle
        history.append({'time': list(time), 'position': list(pos)})
    return history
