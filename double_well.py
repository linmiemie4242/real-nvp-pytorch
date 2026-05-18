"""
2D Double-Well Potential – Overdamped Langevin Equation
========================================================

Potential:
    V(x) = (x1^2 - 1)^2 + x2^2 / 2

SDE (overdamped Langevin):
    dX = -grad V(X) dt + sqrt(2/beta) dW,    X in R^2

Parameters
----------
beta      : inverse temperature  = 2
grad V(x) : (4 x1 (x1^2 - 1), x2)
W         : 2-D standard Brownian motion

Fokker-Planck equation
----------------------
    dp/dt = div[ grad V(x) p ] + (1/beta) Delta p

Stationary distribution (Gibbs measure)
----------------------------------------
    p_inf(x) = Z^{-1} exp( -beta V(x) )

No closed-form time-dependent analytical solution exists for this model.
"""

import numpy as np

# ── Model parameters ──────────────────────────────────────────────────────────
DIM = 2           # spatial dimension
NOISE_DIM = 2     # dimension of the Wiener process

BETA = 2.0        # inverse temperature (kT = 1/beta = 0.5)

# Initial condition: X0 ~ N(X0_MEAN, X0_COV)
# Start localised near the origin
X0_MEAN = np.array([0.0, 0.0])
X0_COV = 1.0 * np.eye(DIM)


# ── Model functions ───────────────────────────────────────────────────────────

def potential(x: np.ndarray) -> np.ndarray:
    """
    V(x) = (x1^2 - 1)^2 + x2^2 / 2.

    Parameters
    ----------
    x : (..., 2)

    Returns
    -------
    V : (...)
    """
    return (x[..., 0] ** 2 - 1.0) ** 2 + x[..., 1] ** 2 / 2.0


def grad_potential(x: np.ndarray) -> np.ndarray:
    """
    grad V(x) = ( 4 x1 (x1^2 - 1),  x2 ).

    Parameters
    ----------
    x : (..., 2)

    Returns
    -------
    gV : (..., 2)
    """
    gV = np.empty_like(x)
    gV[..., 0] = 4.0 * x[..., 0] * (x[..., 0] ** 2 - 1.0)
    gV[..., 1] = x[..., 1]
    return gV


def drift(x: np.ndarray) -> np.ndarray:
    """
    Drift  f(x) = -grad V(x).

    Parameters
    ----------
    x : (..., 2)

    Returns
    -------
    f : (..., 2)
    """
    return -grad_potential(x)


def diffusion(x: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """
    Constant diffusion matrix  g = sqrt(2/beta) * I_2.

    Returns
    -------
    g : (2, 2)
    """
    return np.sqrt(2.0 / BETA) * np.eye(NOISE_DIM)


def div_drift(x: np.ndarray) -> np.ndarray:
    """
    Divergence of the drift: div f(x) = -div grad V(x).

    div grad V = 4(3 x1^2 - 1) + 1 = 12 x1^2 - 3
    => div f   = -(12 x1^2 - 3)

    Parameters
    ----------
    x : (..., 2)

    Returns
    -------
    divf : (...)
    """
    return -(12.0 * x[..., 0] ** 2 - 3.0)


# ── Stationary distribution ───────────────────────────────────────────────────

def stationary_unnorm(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Unnormalised stationary density  exp(-beta V(x)).

    Parameters
    ----------
    x1, x2 : arrays of the same shape (e.g. from np.meshgrid)

    Returns
    -------
    p : array of the same shape as *x1* (not normalised)
    """
    x = np.stack([x1.ravel(), x2.ravel()], axis=-1)
    log_p = -BETA * potential(x)
    p = np.exp(log_p - log_p.max())      # shift for numerical stability
    return p.reshape(x1.shape)


def stationary_pdf(x1: np.ndarray, x2: np.ndarray,
                   dx1: float, dx2: float) -> np.ndarray:
    """
    Normalised stationary density on a grid.

    Parameters
    ----------
    x1, x2 : meshgrid arrays
    dx1, dx2 : grid spacings used for the trapezoid normalisation
    """
    p = stationary_unnorm(x1, x2)
    p /= p.sum() * dx1 * dx2
    return p
