"""
Two-Component Gaussian Mixture Model (GMM) Fitting Module
Resolves bimodal population distributions (e.g., primary vs secondary precipitates).
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.mixture import GaussianMixture

def fit_two_component_gmm(
    diameters_nm: np.ndarray,
    weights: Optional[np.ndarray] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fits a 2-component Gaussian Mixture Model to particle diameters.
    Can be weighted by particle area to capture the area-weighted distribution modes.

    Parameters
    ----------
    diameters_nm : np.ndarray
        Array of equivalent circular diameters.
    weights : np.ndarray, optional
        Sample weights (e.g., particle areas).
    random_state : int
        Seed for reproducibility (default 42).

    Returns
    -------
    Dict[str, Any]
        Dictionary with fitted means (mu1, mu2), standard deviations (std1, std2),
        and component population proportions (weight1, weight2).
    """
    if len(diameters_nm) < 10:
        return {}

    X = diameters_nm.reshape(-1, 1)

    # Scikit-learn GMM sample weighting
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=random_state
    )

    if weights is not None:
        norm_weights = weights / np.sum(weights)
        # Resample based on weights for GMM
        indices = np.random.choice(len(X), size=min(10000, len(X) * 2), p=norm_weights, replace=True)
        X_fit = X[indices]
    else:
        X_fit = X

    gmm.fit(X_fit)

    means = gmm.means_.flatten()
    covars = gmm.covariances_.flatten()
    stds = np.sqrt(covars)
    comp_weights = gmm.weights_.flatten()

    # Sort components so component 1 is smaller mode, component 2 is larger mode
    sort_idx = np.argsort(means)

    return {
        "mu1_nm": float(means[sort_idx[0]]),
        "std1_nm": float(stds[sort_idx[0]]),
        "weight1": float(comp_weights[sort_idx[0]]),
        "mu2_nm": float(means[sort_idx[1]]),
        "std2_nm": float(stds[sort_idx[1]]),
        "weight2": float(comp_weights[sort_idx[1]]),
    }
