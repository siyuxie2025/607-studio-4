import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, r_squared

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # This test should initially fail
    

# TODO: Add your unit tests here

def test_bootstrap_sample_invalidg_inputs():
    """
    Test that bootstrap_sample raises errors on invalid inputs
    """
    if len(X) != len(y):
        raise ValueError("Number of samples in X and y must be the same.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer.")
    if not callable(compute_stat):
        raise ValueError("compute_stat must be a callable function.")
    
    pytest.raises(ValueError, bootstrap_sample, X, y, compute_stat, n_bootstrap=-10)
    pytest.raises(ValueError, bootstrap_sample, X, y, compute_stat, n_bootstrap=0)
    pytest.raises(ValueError, bootstrap_sample, X, y, compute_stat, n_bootstrap=1.5)

    pass

def test_bootstrap_ci_invalid_inputs():
    """
    Test that bootstrap_ci raises errors on invalid inputs
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    
    pytest.raises(ValueError, bootstrap_ci, np.array([1,2,3]), alpha=-0.1)
    pytest.raises(ValueError, bootstrap_ci, np.array([1,2,3]), alpha=0)
    pytest.raises(ValueError, bootstrap_ci, np.array([1,2,3]), alpha=1)
    pytest.raises(ValueError, bootstrap_ci, np.array([1,2,3]), alpha=1.5)

    pass 

def test_r_squared_invalid_inputs():
    """
    Test that r_squared raises errors on invalid inputs
    """
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2, 3])  # Mismatched length

    with pytest.raises(ValueError, match = 'mismatched length'):
        r_squared(X, y)

    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])  # Valid length but not enough samples

    with pytest.raises(ValueError, match = 'not enough samples'):
        r_squared(X, y)

    X = np.array([[1], [2], [3]])  # Not enough features
    y = np.array([1, 2, 3])

    with pytest.raises(ValueError, match = 'not enough features'):
        r_squared(X, y)


    pass