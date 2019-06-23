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
            # Robust(Wasserstein(0.5)),
            # Robust(Wasserstein(2)),
            Robust(Ellipsoid(0.1312)),
            # Robust(Ellipsoid(0.52)),
            # Robust(Ellipsoid(0.7)),
            # Robust(Ellipsoid(2)),
            Robust(Ellipsoid(0.1)),
            Robust(Ellipsoid(0.2)),
            Robust(Ellipsoid(0.3)),
            Robust(Ellipsoid(0.4)),
            Robust(Interval(variance=0.0138)),
        ],  # max 12 models (no further colors or shapes are defined
        Options(
            number_of_paths=100,
            number_of_sims=100,
            plot_hist=True,
            do_simulation=False,
            evaluate_all=True,
            evaluate_inner=True,
            evaluate_outer=True,
            sample_var=0.5,
            sample_amount=2000,
            sample_method = "normal",  # normal, uniform, monte carlo
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
