from Evaluator import Evaluator
from Options import Options, LoggingBehavior
from mdptoolbox.Robust import *
from mdptoolbox.mdp import ValueIteration

from Problem import Problem


"""
default configuration, runs the forest problem on some models
"""


def run_default():
    # get tk_low and tk_up for the interval model
    forest10 = Problem.get_forest_problem(S=10, discount_factor=0.9, r1=10, r2=2, p=0.05)
    forest30 = Problem.get_forest_problem(S=30, discount_factor=0.9, r1=10, r2=2, p=0.05)
    forest80 = Problem.get_forest_problem(S=80, discount_factor=0.9, r1=10, r2=2, p=0.05)
    forest100 = Problem.get_forest_problem(S=100, discount_factor=0.9, r1=10, r2=2, p=0.05)
    random10 = Problem.get_random_problem(10, 10, 0.9)
    tk_low10, tk_up10 = Interval.compute_interval(forest10.transition_kernel, 0.0138)
    rtk_low, rtk_up = Interval.compute_interval(random10.transition_kernel, 0.0138)
    # tk_low = (tk-0.5).clip(min=0)
    # tk_up = (tk+0.5).clip(max=1)

    # problems can also be supplied as a list
    evaluator = Evaluator(
        [
            forest10,
            random10,
            forest30,
            forest80,
            forest100
        ],
        [
            # Robust(Wasserstein(0.03)),
            Robust(Wasserstein(0.07)),
            # Robust(Wasserstein(0.17)),
            # Robust(Wasserstein(0.5)),
            # Robust(Wasserstein(2)),
            Robust(Ellipsoid(0.1312)),
            # Robust(Ellipsoid(0.52)),
            # Robust(Ellipsoid(0.7)),
            # Robust(Ellipsoid(2)),
            ValueIteration,
            # Robust(Likelihood(0.1, 0.001)),
            # Robust(Likelihood(0.3, 0.001)),
            Robust(Likelihood(0.5, 0.001)),
            # Robust(Likelihood(0.6, 0.001)),
            # Robust(Likelihood(0.7, 0.001)),
            # Robust(Likelihood(0.9, 0.001)),
            # Robust(Likelihood(1, 0.001)),
            # Robust(Likelihood(1.2, 0.001)),
            # Robust(Likelihood(1.5, 0.001)),
            # Robust(Likelihood(1.7, 0.001)),
            # Robust(Likelihood(1.9, 0.001)),
            # Robust(Likelihood(2.1, 0.001)),
            # Robust(Likelihood(2.4, 0.001)),
            # Robust(Likelihood(2.8, 0.001)),
            # Robust(Likelihood(3, 0.001)), # range 1.5 - 0ish
            Robust(Interval(variance=0.0138)),
        ],  # max 12 models (no further colors or shapes are defined
        Options(
            number_of_paths=2000,
            number_of_sims=1000,
            plot_hist=True,
            do_simulation=False,
            evaluate_all=True,
            evaluate_inner=True,
            evaluate_outer=True,
            sample_var=0.5,
            sample_amount=2000,
            sample_method = "normal", #normal, uniform, monte carlo
            monte_carlo_sampling_init_count_value = 1,
            monte_carlo_sampling_random_samples = 10,
            use_problem_set_for_policy=False,
            non_robust_actions=[],  # replace with [1] if for action there should be no robustness
            variance_scaling=True,
            variance_lower=0,
            variance_upper=1,
            logging_behavior=LoggingBehavior.TABLE,

        ))
    evaluator.run()


run_default()
