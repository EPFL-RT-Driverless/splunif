#  Copyright (c) 2022. Tudor Oancea, EPFL Racing Team Driverless
# A very simple package that constructs 2D uniform splines, whose continuous parameters
# correspond to the arc length of the spline.

from typing import Tuple

import numpy as np
from scipy.integrate import quadrature
from scipy.interpolate import CubicSpline
from scipy.optimize import bisect


def uniform_points_and_breaks(
    ref_points: np.ndarray, nbr_interpolation_points: int = None, periodic: bool = True
):
    """
    Fits a spline using the give reference points and computes new breaks and
    interpolation points on it that will yield a uniform spline (i.e. whose continuous
    parameter corresponds to the arc length).

    The total length of the curve corresponds to the last returned break point.

    :param ref_points: array of initial reference points, shape=(2,N)
    :param nbr_interpolation_points: the number of generated breaks / interpolation points.
    If none specified, will correspond to 2*N.
    :param periodic: true if the spline is closed, false otherwise.
    :return: new interpolation points (np.ndarray of shape (2,nbr_interpolation_points))
     and the break points to use to fit the uniform spline.
    """
    N = ref_points.shape[1] - 1
    t = np.linspace(0.0, N, N + 1, dtype=float)
    x_ref = CubicSpline(
        t,
        ref_points[0, :],
        bc_type="periodic" if periodic else "not-a-knot",
    )
    y_ref = CubicSpline(
        t,
        ref_points[1, :],
        bc_type="periodic" if periodic else "not-a-knot",
    )
    length = lambda t1, t2: quadrature(
        lambda u: np.sqrt(x_ref(u, 1) ** 2 + y_ref(u, 1) ** 2),
        t1,
        t2,
    )[0]

    # Step 1 : find the lengths of each segment of the original curve =================
    l = np.zeros(N)
    for i in range(N):
        l[i] = length(t[i], t[i + 1])

    s = np.concatenate(([0.0], np.cumsum(l)))
    L = s[-1]
    print("L = ", L)

    # Step 2 : find the uniformization points ========================================
    M = nbr_interpolation_points
    t_tilde = np.zeros(M + 1)
    t_tilde[-1] = N
    lam = L / M * np.ones(M)
    sigma = np.concatenate(([0], np.cumsum(lam)))
    assert (
        np.max(np.abs(sigma - L / M * np.arange(M + 1))) < 1e-10
    ), "sigma is not well computed"

    for j in range(M - 1):
        i = np.searchsorted(s, sigma[j + 1], side="right")
        obj = (
            lambda upper_bound: length(t[i - 1], upper_bound) - sigma[j + 1] + s[i - 1]
        )
        t_tilde[j + 1] = bisect(
            obj,
            t[i - 1],
            t[i],
        )

    # step 3 : construct the new re-parametrized spline ==============================
    new_points = np.zeros((2, M + 1))
    for j in range(M + 1):
        new_points[0, j] = x_ref(t_tilde[j])
        new_points[1, j] = y_ref(t_tilde[j])

    return new_points, sigma


def uniform_spline(
    ref_points: np.ndarray, nbr_interpolation_points: int = None, periodic: bool = True
) -> Tuple[CubicSpline, CubicSpline, float]:
    """
    Uses the function uniform_points_and_breaks() to compute and return two scipy
    CubicSpline and the length of the spline.

    :param ref_points: array of initial reference points, shape=(2,N)
    :param nbr_interpolation_points: the number of generated breaks / interpolation points.
    If none specified, will correspond to 2*N.
    :param periodic: true if the spline is closed, false otherwise.
    :return: reference splines in X and Y, and total length of the spline.
    """
    new_ref_points, sigma = uniform_points_and_breaks(
        ref_points=ref_points,
        nbr_interpolation_points=nbr_interpolation_points,
        periodic=periodic,
    )
    x_ref = CubicSpline(
        sigma,
        new_ref_points[0, :],
        bc_type="periodic" if periodic else "not-a-knot",
    )
    y_ref = CubicSpline(
        sigma,
        new_ref_points[1, :],
        bc_type="periodic" if periodic else "not-a-knot",
    )
    return x_ref, y_ref, sigma[-1]
