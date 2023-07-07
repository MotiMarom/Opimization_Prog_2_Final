# Moti Marom
# ID 025372830

# test_constrained_min.py
import numpy as np
from constrained_min import interior_pt
from examples import test_qp
from examples import test_lp
from utils import plot_results_linear
from utils import plot_results_quadratic

# Get the requested function to analyze from user
print('Hello!')
print('Please pick a function for analysis from the following:')
print('1-quadratic, 2-linear')
function_index = input('type a single number between [1, 2]:')
function_index = int(function_index)

if function_index == 1:
    print('You chose 1: quadratic')
    func_name = 'quadratic'
    bt = True
    func2min = test_qp
    x0 = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    results_newton = interior_pt(func2min, x0, backtrack=bt, m=4, t=1.0, miu=10, eps_barrier=1e-5, eps_newton=1e-5)

    # Plot results
    x_track = results_newton[2]
    f_track = results_newton[3]
    outer_iter = np.arange(len(f_track))
    x_limits = np.array([-2, 2])
    y_limits = np.array([-2, 2])
    z_limits = np.array([-2, 2])
    plot_results_quadratic(func2min, x_track, f_track, x_limits, y_limits, z_limits, func_name)

elif function_index == 2:
    print('You chose 2: linear')
    func_name = 'linear'
    bt = False
    func2min = test_lp
    x0 = np.array([0.5, 0.75], dtype=np.float64)
    results_newton = interior_pt(func2min, x0, backtrack=bt, m=4, t=1.0, miu=10, eps_barrier=1e-5, eps_newton=1e-5)

    # Plot results
    x_track = results_newton[2]
    f_track = results_newton[3]
    outer_iter = np.arange(len(f_track))
    x_limits = np.array([-1, 3])
    y_limits = np.array([-1, 3])
    plot_results_linear(func2min, x_track, f_track, x_limits, y_limits, func_name)

else:
    print("You chose {} where it should be an integer number between 1-2. please rerun and try again."
          .format(function_index))


print('End of {} analysis'.format(func_name))

print('End of running.')


