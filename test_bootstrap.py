import pytest

import numpy as np

from bootstrap import bootstrap_sample, bootstrap_ci, r_squared

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # This test should initially fail
    

# TODO: Add your unit tests here

class TestBootstrapSample:
    """Test suite for bootstrap functions"""
    def compute_stat(self, X, y):
        return np.mean(y)
    
    def test_bootstrap_sample_happy_path(self):
        """Test basic functionality of bootstrap_sample."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])

        stats = bootstrap_sample(X, y, self.compute_stat, n_bootstrap=1000)

        assert len(stats) == 1000
        assert abs(np.mean(stats) - np.mean(y)) < 0.1  # Mean should be close to original mean

    def test_bootstrap_sample_invalid_n_bootstrap(self):
        """Test that bootstrap_sample raises errors on invalid n_bootstrap."""

        with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
            bootstrap_sample([1, 2], [1, 2], self.compute_stat, n_bootstrap=-10)


    def test_bootstrap_sample_invalid_inputs_non_array(self):
        """Test that bootstrap_sample raises errors on non-array inputs."""
        X = [[1], [2], [3], [4], [5]]
        y = [1, 2, 3, 4, 5]
        
        with pytest.raises(ValueError, match="X and y must be array-like"):
            bootstrap_sample("invalid", y, self.compute_stat, n_bootstrap=100)
        with pytest.raises(ValueError, match="X and y must be array-like"):
            bootstrap_sample(X, "invalid", self.compute_stat, n_bootstrap=100)

    def test_bootstrap_sample_invalid_inputs_mismatched_length(self):
        """Test that bootstrap_sample raises errors on mismatched lengths."""
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2])  # Mismatched length

        with pytest.raises(ValueError, match="X and y must have the same length"):
            bootstrap_sample(X, y, self.compute_stat, n_bootstrap=100)
