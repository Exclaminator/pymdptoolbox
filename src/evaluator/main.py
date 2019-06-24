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
    def forest(size):
        return Problem.get_forest_problem(S=size, discount_factor=0.9, r1=10, r2=2, p=0.05)

    def random(size, actions=5):
        return Problem.get_random_problem(S=size, A=actions, discount_factor=0.9)

    # problems can also be supplied as a list
    evaluator = Evaluator(
        [
            forest(5),
            forest(10),
            forest(30),
            random(5),
            random(10),
            random(30)
        ],
        [
            Robust(Wasserstein(0.1658)),
            Robust(Likelihood(4.07, 0.001)),
            Robust(Ellipsoid(0.345)),
            Robust(Interval(variance=0.105)),
            ValueIteration
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
