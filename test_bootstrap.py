import pytest

import numpy as np

from bootstrap import bootstrap_sample, bootstrap_ci, r_squared

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # This test should initially fail
    

# TODO: Add your unit tests here

class TestBootstrap:
    """Test suite for bootstrap functions"""
    
    def test_bootstrap_sample_happy_path(self):
        """Test basic functionality of bootstrap_sample."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])
        
        def compute_stat(X, y):
            return np.mean(y)
    
        stats = bootstrap_sample(X, y, compute_stat, n_bootstrap=1000)

        assert len(stats) == 1000
        assert abs(np.mean(stats) - np.mean(y)) < 0.1  # Mean should be close to original mean
    
    def test_bootstrap_ci_happy_path(self):
        """Test basic functionality of bootstrap_ci."""
        bootstrap_stats = np.random.normal(loc=0, scale=1, size=1000)
        ci_lower, ci_upper = bootstrap_ci(bootstrap_stats, alpha=0.05)
        
        assert ci_lower < ci_upper
        assert abs(ci_lower) < 3  # Rough bounds for normal distribution
        assert abs(ci_upper) < 3

    def test_r_squared_happy_path(self):
        """Test basic functionality of r_squared."""
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
        y = np.array([1, 2, 3, 4, 5])
        
        r2 = r_squared(X, y)
        assert abs(r2 - 1.0) < 1e-10  # Perfect fit


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