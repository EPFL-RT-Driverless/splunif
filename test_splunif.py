#  Copyright (c) 2022. Tudor Oancea, EPFL Racing Team Driverless

from matplotlib import pyplot as plt
from scipy.integrate import quadrature

from splunif import uniform_points_and_breaks
import numpy as np
from scipy.interpolate import CubicSpline

if __name__ == "__main__":
    data = np.loadtxt("fs_track.csv", delimiter=",", skiprows=1)
    left_cones = data[:, :2].T
    right_cones = data[:, 2:4].T
    center_line_points = data[:, 4:6].T

    # declare uniform spline
    new_points, sigma = uniform_points_and_breaks(
        ref_points=center_line_points,
        nbr_interpolation_points=2 * center_line_points.shape[1],
    )
    M = sigma.size - 1
    L = sigma[-1]
    new_x_cl = CubicSpline(
        sigma,
        new_points[0, :],
        bc_type="periodic",
    )
    new_y_cl = CubicSpline(
        sigma,
        new_points[1, :],
        bc_type="periodic",
    )
    new_length = lambda new_t1, new_t2: quadrature(
        lambda u: np.sqrt(new_x_cl(u, 1) ** 2 + new_y_cl(u, 1) ** 2),
        new_t1,
        new_t2,
    )[0]
    lam = np.zeros(M)
    for j in range(M):
        lam[j] = new_length(sigma[j], sigma[j + 1])
    computed_sigma = np.concatenate(([0.0], np.cumsum(lam)))

    # declare naive spline
    N = center_line_points.shape[1] - 1
    t = np.linspace(0.0, L, N + 1)
    x_cl = CubicSpline(
        t,
        center_line_points[0, :],
        bc_type="periodic",
    )
    y_cl = CubicSpline(
        t,
        center_line_points[1, :],
        bc_type="periodic",
    )
    length = lambda t1, t2: quadrature(
        lambda u: np.sqrt(x_cl(u, 1) ** 2 + y_cl(u, 1) ** 2),
        t1,
        t2,
    )[0]
    l = np.zeros(N)
    for i in range(N):
        l[i] = length(t[i], t[i + 1])
    s = np.concatenate(([0.0], np.cumsum(l)))

    # plot the two reference path to make sure they are the same
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(
        x_cl(np.linspace(0.0, N, 4 * N, dtype=float)),
        y_cl(np.linspace(0.0, N, 4 * N, dtype=float)),
        "r-",
    )
    plt.plot(
        new_x_cl(np.linspace(0.0, L, 4 * M, dtype=float)),
        new_y_cl(np.linspace(0.0, L, 4 * M, dtype=float)),
        "g-",
    )
    plt.legend(["cl", "new_cl"])
    plt.plot(left_cones[0, :], left_cones[1, :], "b+")
    plt.plot(right_cones[0, :], right_cones[1, :], "y+")
    plt.axis("equal")
    plt.grid("on")
    plt.title("map of original and uniformized curve")

    # plot the new arc length vs the old one
    plt.subplot(1, 2, 2)
    plt.plot(t, s, "r-")
    plt.plot(sigma, computed_sigma, "g-")
    # plt.plot([0.0, L], [0.0, L], "b-")
    plt.legend(["s", "sigma", "ref"])
    plt.axis("equal")
    plt.grid("on")

    plt.show()
