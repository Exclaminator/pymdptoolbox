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
    forest10 = Problem.get_forest_problem(S=10, discount_factor=0.9, p=0.1)
    # forest30 = Problem.get_forest_problem(S=30, discount_factor=0.9, r1=10, r2=2, p=0.05)
    # forest80 = Problem.get_forest_problem(S=80, discount_factor=0.9, r1=10, r2=2, p=0.05)
    # forest100 = Problem.get_forest_problem(S=100, discount_factor=0.9, r1=10, r2=2, p=0.05)
    random = Problem.get_random_problem(10, 5, 0.9, seed=40)
    # tk_low10, tk_up10 = Interval.compute_interval(forest10.transition_kernel, 0.0138)
    # rtk_low, rtk_up = Interval.compute_interval(random10.transition_kernel, 0.0138)
    # tk_low = (tk-0.5).clip(min=0)
    # tk_up = (tk+0.5).clip(max=1)

    # problems can also be supplied as a list
    evaluator = Evaluator(
        [
            forest10,
            # random,
            # forest30,
            # forest80,
            # forest100
        ],
        [
            # Robust(Wasserstein(0.05)),
            # Robust(Wasserstein(0.15)),
            Robust(Wasserstein(0.142)),
            Robust(Wasserstein(0.2)),
            # Robust(Wasserstein(0.25)),
            # Robust(Wasserstein(0.3)),
            # Robust(Wasserstein(0.4)),
            # Robust(Wasserstein(0.165)),
            # Robust(Wasserstein(0.167)),

            # Robust(Wasserstein(0.164)),
            Robust(Likelihood(0.1, 0.01)),
            # Robust(Likelihood(0.2, 0.01)),
            Robust(Ellipsoid(0.21)),
            # Robust(Ellipsoid(0.22)),
            # Robust(Ellipsoid(0.31)),

            # Robust(Interval(variance=0.088)),
            Robust(Interval(variance=0.079)),

            # Robust(Interval(variance=0.09)),
            # Robust(Interval(variance=0.0918)),
            # Robust(Interval(variance=0.092)),
            # Robust(Interval(variance=0.097)),
            # Robust(Interval(variance=0.098)),
            # Robust(Interval(variance=0.11)),
            ValueIteration
            # Robust(Interval(rtk_low, rtk_up))
        ],  # max 12 models (no further colors or shapes are defined
        Options(
            number_of_paths=1000,
            number_of_sims=100,
            plot_hist=True,
            do_simulation=False,
            evaluate_all=True,
            evaluate_inner=True,
            evaluate_outer=True,
            sample_var=1,
            sample_amount=1000,
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
