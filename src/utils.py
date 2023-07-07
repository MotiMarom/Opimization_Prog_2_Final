# Moti Marom
# ID 025372830

# utils.py

import numpy as np
import matplotlib.pyplot as plt


def plot_linear_feasible_set(x, y):
    c1 = lambda a, b: - a - b + 1
    c2 = lambda a, b: b - 1
    c3 = lambda a, b: a - 2
    c4 = lambda a, b: -b
    plt.imshow(
        ((c1(x, y) <= 0) & (c2(x, y) <= 0) & (c3(x, y) <= 0) & (c4(x, y) <= 0)).astype(int),
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin='lower',
        cmap='inferno'
    )


def plot_linear_contour(obj_func, x_limits, y_limits):

    # Meshgrid that covers the relevant zone
    x0 = np.linspace(x_limits[0], x_limits[1], 100)
    x1 = np.linspace(y_limits[0], y_limits[1], 100)
    mesh_x0, mesh_x1 = np.meshgrid(x0, x1)
    n_rows, n_cols = mesh_x0.shape
    func_mesh_x = np.array(np.zeros((n_rows, n_cols)), dtype=np.float64)

    for r in range(n_rows):
        for c in range(n_cols):
            x_in_to_obj_f = np.array([mesh_x0[r, c], mesh_x1[r, c]], dtype=np.float64)
            func_x, g_NA, h_NA = obj_func(x_in_to_obj_f, t=0)
            func_mesh_x[r, c] = func_x

    # Plot
    ContourSurface = plt.contour(x0, x1, func_mesh_x)
    plt.clabel(ContourSurface, fmt='%1.2f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour of $f(x)$ along with the Feasible Set and Iteration Path')
    plt.show()


def plot_learning_curve(fx, func_name):

    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(len(fx) - 1, fx[-1], c='r')
    ax.plot(fx, c='b')
    ax.set_title('$f(x) = f$({}) log-barrier method'.format(func_name))
    ax.set_xlabel('$number of outer iterations$')
    ax.set_ylabel('$f(x)$')
    plt.show()


def plot_results_linear(func2min, x_track, f_track, x_limits, y_limits,func_name):

    plt.figure(figsize=(10, 10))
    plt.plot(x_track[:, 0], x_track[:, 1], 'b')
    plt.scatter(x_track[-1][0], x_track[-1][1], color='r')

    x0 = np.linspace(x_limits[0], x_limits[1], 200)
    x1 = np.linspace(y_limits[0], y_limits[1], 200)
    x, y = np.meshgrid(x0, x1)

    plot_linear_feasible_set(x, y)
    plot_linear_contour(func2min, x_limits, y_limits)
    plot_learning_curve(f_track, func_name)


def plot_quadratic_feasible_set(x, y):
    c1 = lambda a, b: - a
    c2 = lambda a, b: - b
    plt.imshow(
        ((c1(x, y) <= 0) & (c2(x, y) <= 0)).astype(int),
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin='lower',
        cmap='inferno'
    )


def plot_quadratic_contour(obj_func, x_limits, y_limits, z_limits):

    # Meshgrid that covers the relevant zone
    X = np.linspace(x_limits[0], x_limits[1], 100)
    Y = np.linspace(y_limits[0], y_limits[1], 100)
    Z = np.linspace(z_limits[0], z_limits[1], 100)
    mesh_x, mesh_y, mesh_z = np.meshgrid(X, Y, Z)
    n_x, n_y, n_z = mesh_x.shape
    func_mesh_xyz = np.array(np.zeros((n_x, n_y, n_z)), dtype=np.float64)

    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                x_in_to_obj_f = np.array([mesh_x[x, y, z], mesh_y[x, y, z], mesh_z[x, y, z]], dtype=np.float64)
                func_x, g_NA, h_NA = obj_func(x_in_to_obj_f, t=0)
                func_mesh_xyz[x, y, z] = func_x

    # Plot
    ContourSurface = plt.contour(X, Y, func_mesh_xyz[:, :, 0])
    plt.clabel(ContourSurface, fmt='%1.2f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour of $f(x)$ along with the Feasible Set and Iteration Path')
    plt.show()


def plot_results_quadratic(func2min, x_track, f_track, x_limits, y_limits, z_limits, func_name):

    plt.figure(figsize=(10, 10))
    plt.plot(x_track[:, 0], x_track[:, 1], 'b')
    plt.scatter(x_track[-1][0], x_track[-1][1], color='r')

    X = np.linspace(x_limits[0], x_limits[1], 100)
    Y = np.linspace(y_limits[0], y_limits[1], 100)
    x, y = np.meshgrid(X, Y)

    plot_quadratic_feasible_set(x, y)
    plot_quadratic_contour(func2min, x_limits, y_limits, z_limits)
    plot_learning_curve(f_track, func_name)


