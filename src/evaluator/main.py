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
    forest = Problem.get_forest_problem(S=10, discount_factor=0.9, r1=10, r2=2, p=0.05)
    random = Problem.get_random_problem(10, 10, 0.9)
    tk_low, tk_up = Interval.compute_interval(forest.transition_kernel, 0.0138)
    rtk_low, rtk_up = Interval.compute_interval(random.transition_kernel, 0.0138)
    # tk_low = (tk-0.5).clip(min=0)
    # tk_up = (tk+0.5).clip(max=1)

    # problems can also be supplied as a list
    evaluator = Evaluator(
        [forest, random],
        [
            # Robust(Wasserstein(0.03)),
            # Robust(Wasserstein(0.07)),
            # Robust(Wasserstein(0.17)),
            # Robust(Wasserstein(0.5)),
            # Robust(Wasserstein(2)),
            # Robust(Ellipsoid(0.1312)),
            # ValueIteration,
            Robust(Likelihood(0.1, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(0.3, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(0.5, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(0.6, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(0.7, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(0.9, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(1, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(1.2, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(1.5, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(1.7, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(1.9, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(2.1, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(2.4, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(2.8, 0.001)), # range 1.5 - 0ish
            Robust(Likelihood(3, 0.001)), # range 1.5 - 0ish
            # Robust(Interval(tk_low, tk_up)),
            # Robust(Interval(rtk_low, rtk_up))
        ],
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
            sample_uniform=False,
            non_robust_actions=[], # replace with [1] if for action there should be no robustness
            variance_scaling=True,
            variance_lower=0,
            variance_upper=1,
            logging_behavior=LoggingBehavior.TABLE
        ))
    evaluator.run()


run_default()
