import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# Go up one level to the root, then dig into the 'src' folder
root_dir = Path(__file__).parent.parent
src_dir = root_dir / "src"

# Tell Python to look inside the 'src' folder for your library
sys.path.insert(0, str(src_dir))

@pytest.fixture
def default_opts():
    """
    gets the default options for vectfit. 
    """
    # Adjust this import based on where your vectfit3.py actually lives
    # If it's in the same root directory, this will work.
    from vectfit3 import opts 
    return opts.copy()

@pytest.fixture
def generate_synthetic_fs():
    """
    Factory fixture: Generates an F(s) frequency response.
    Supports Scalar (1D) and Vector (2D) responses.
    """
    def _generate(s, expected_poles, expected_residues, D=None, E=None):
        # Determine if we are doing scalar or vector fitting
        # expected_residues shape: (Nc, n_poles) or (n_poles,)
        if expected_residues.ndim == 1:
            F = np.zeros_like(s, dtype=np.complex128)
            for p, r in zip(expected_poles, expected_residues):
                F += r / (s - p)
            if D is not None: F += D
            if E is not None: F += s * E
        else:
            # Vector case: Nc x N
            Nc = expected_residues.shape[0]
            N = len(s)
            F = np.zeros((Nc, N), dtype=np.complex128)
            for i in range(Nc):
                for p, r in zip(expected_poles, expected_residues[i]):
                    F[i, :] += r / (s - p)
                if D is not None: F[i, :] += D[i]
                if E is not None: F[i, :] += s * E[i]
        return F
    return _generate



@pytest.fixture
def load_vectfit_csv():
    """
    Clean CSV loader for test datasets. Expects files to be in a 'data' subfolder next to this conftest.py.
    """
    data_path = Path(__file__).parent / "data"

    def _loader(filename):
        file_path = data_path / filename
        if not file_path.exists():
            pytest.fail(f"Dataset {filename} not found in {data_path}")
        
        # We use pandas to read, then convert to numpy for speed
        df = pd.read_csv(file_path)
        return df.to_numpy()

    return _loader