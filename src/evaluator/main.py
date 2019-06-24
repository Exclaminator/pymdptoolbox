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
    # forest30 = Problem.get_forest_problem(S=30, discount_factor=0.9, r1=10, r2=2, p=0.05)
    # forest80 = Problem.get_forest_problem(S=80, discount_factor=0.9, r1=10, r2=2, p=0.05)
    # forest100 = Problem.get_forest_problem(S=100, discount_factor=0.9, r1=10, r2=2, p=0.05)
    random = Problem.get_random_problem(10, 10, 0.9, seed=50)
    # tk_low10, tk_up10 = Interval.compute_interval(forest10.transition_kernel, 0.0138)
    # rtk_low, rtk_up = Interval.compute_interval(random10.transition_kernel, 0.0138)
    # tk_low = (tk-0.5).clip(min=0)
    # tk_up = (tk+0.5).clip(max=1)

    # problems can also be supplied as a list
    evaluator = Evaluator(
        [
            # forest10,
            random,
            # forest30,
            # forest80,
            # forest100
        ],
        [
            # Robust(Wasserstein(0.06)),
            # Robust(Wasserstein(0.165)),
            # Robust(Wasserstein(0.1655)),
            Robust(Wasserstein(0.1658)),
            # Robust(Wasserstein(0.166)),
            # Robust(Wasserstein(0.167)),
            # Robust(Wasserstein(0.168)),
            # Robust(Wasserstein(0.169)),
            # Robust(Likelihood(3.9, 0.001)),
            # Robust(Likelihood(4, 0.001)),
            Robust(Likelihood(4.07, 0.001)),
            # Robust(Likelihood(4.2, 0.001)),
            # Robust(Likelihood(4.3, 0.001)),
            # Robust(Likelihood(4.4, 0.001)),
            # Robust(Ellipsoid(0.1)),
            # Robust(Ellipsoid(0.15)),
            # Robust(Ellipsoid(0.33)),
            Robust(Ellipsoid(0.345)),
            # Robust(Ellipsoid(0.35)),
            # Robust(Ellipsoid(0.4)),
            # Robust(Ellipsoid(0.45)),
            # Robust(Interval(variance=0.1)),
            Robust(Interval(variance=0.105)),
            # Robust(Interval(variance=0.08)),
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
            sample_var=0.1,
            sample_amount=10000,
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
