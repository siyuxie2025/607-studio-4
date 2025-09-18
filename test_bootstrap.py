import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, r_squared

class TestBootstrap:
    """
    Test for bootstrap functions
    """
    def test_r_squared_function_happy_path(self):
        """
        Test the basic functionality of r_squared
        """
        # Simple linear relationship
        X = np.column_stack([np.ones(5), [1, 2, 3, 4, 5]])
        y = np.array([2, 4, 6, 8, 10])
        
        r2 = r_squared(X, y)
        assert abs(r2 - 1.0) < 1e-10, "Should have R² ≈ 1"

    def test_r_squared_function_edge_cases(self):
        """
        Test the r_squared function
        """
        # No relationship (constant y)
        X = np.column_stack([np.ones(5), [1, 2, 3, 4, 5]])
        y_constant = np.array([5, 5, 5, 5, 5])
        r2_constant = r_squared(X, y_constant)
        assert r2_constant < 1e-10, "Constant y should have R² ≈ 0"
        
        # Mismatched lengths
        with pytest.raises(ValueError, match="Number of samples in X and y must be the same."):
            r_squared(np.array([[1, 1], [1, 2]]), np.array([1, 2, 3]))

        # Small sample size
        with pytest.raises(ValueError, match="Number of samples must be greater than number of features."):
            r_squared(np.array([[1, 1], [1, 2]]), np.array([1, 2])) 

        # Feature matrix without intercept
        with pytest.raises(ValueError, match="X must include at least one feature and an intercept."):
            r_squared(np.array([[1], [2], [3]]), np.array([1, 2, 3]))

    
    def test_bootstrap_ci_basic(self):
        """
        Test basic functionality with known data
        """
        # Simple test with known percentiles
        bootstrap_stats = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # For 95% CI (alpha=0.05): 2.5th and 97.5th percentiles
        lower, upper = bootstrap_ci(bootstrap_stats, alpha=0.05)
        
        expected_lower = np.percentile(bootstrap_stats, 2.5)  # 1.225
        expected_upper = np.percentile(bootstrap_stats, 97.5)  # 9.775
        
        assert abs(lower - expected_lower) < 1e-10
        assert abs(upper - expected_upper) < 1e-10
        assert lower < upper, "Lower bound should be less than upper bound"
    
    def test_bootstrap_ci_different_alphas(self):
        """
        Test different confidence levels
        """
        bootstrap_stats = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 90% CI (alpha=0.10): 5th and 95th percentiles
        lower_90, upper_90 = bootstrap_ci(bootstrap_stats, alpha=0.10)
        
        # 99% CI (alpha=0.01): 0.5th and 99.5th percentiles  
        lower_99, upper_99 = bootstrap_ci(bootstrap_stats, alpha=0.01)
        
        # 99% CI should be wider than 90% CI
        width_90 = upper_90 - lower_90
        width_99 = upper_99 - lower_99
        assert width_99 > width_90, "99% CI should be wider than 90% CI"
        
        # Check values
        assert abs(lower_90 - np.percentile(bootstrap_stats, 5.0)) < 1e-10
        assert abs(upper_90 - np.percentile(bootstrap_stats, 95.0)) < 1e-10
    
    def test_bootstrap_ci_edge_cases(self):
        """
        Test edge cases and boundary conditions
        """
        
        # Identical values
        identical_stats = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        lower, upper = bootstrap_ci(identical_stats, alpha=0.05)
        assert lower == upper == 5.0, "CI of identical values should be the single value"
        
        # Single value
        single_stat = np.array([3.14])
        lower, upper = bootstrap_ci(single_stat, alpha=0.05)
        assert lower == upper == 3.14, "CI of single value should be that value"
        
        # Two values
        two_stats = np.array([1.0, 9.0])
        lower, upper = bootstrap_ci(two_stats, alpha=0.05)
        assert lower <= upper, "Even with two values, lower ≤ upper"