
from sklearn.linear_model import LinearRegression

"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""


def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    ....
    """
    X = np.array(X)
    y = np.array(y)
    n = len(y)
    
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Compute the statistic on bootstrap sample
        bootstrap_stats[i] = compute_stat(X_boot, y_boot)
    
    return bootstrap_stats

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI
    
    ....
    """
    lower_bound = np.percentile(bootstrap_stats, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return (lower_bound, upper_bound)

    pass

def R_squared(X, y):
    """
    Calculate R-squared from multiple linear regression.

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)

    Returns
    -------
    float
        R-squared value (between 0 and 1) from OLS
    
    Raises
    ------
    ValueError
        If X.shape[0] != len(y)
    """

    if X.shape[0] != len(y):
        raise ValueError("Number of samples in X and y must be the same.")
    if X.shape[0] <= X.shape[1]:
        raise ValueError("Number of samples must be greater than number of features.")
    if X.shape[1] < 2:
        raise ValueError("X must include at least one feature and an intercept.")
    
    model = LinearRegression().fit(X, y)
    r2_original = model.score(X, y)

    return r2_original
    

    pass