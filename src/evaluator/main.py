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
   # forest = Problem.get_forest_problem(S=10, discount_factor=0.9, r1=10, r2=2, p=0.05)
    forest = Problem.get_random_problem()
    tk_low1, tk_up1 = Interval.compute_interval(forest.transition_kernel, 0.08)
    tk_low2, tk_up2 = Interval.compute_interval(forest.transition_kernel, 0.04)
    tk_low3, tk_up3 = Interval.compute_interval(forest.transition_kernel, 0.2)
    tk_low4, tk_up4 = Interval.compute_interval(forest.transition_kernel, 0.3)
    tk_low5, tk_up5 = Interval.compute_interval(forest.transition_kernel, 0.4)
    tk_low6, tk_up6 = Interval.compute_interval(forest.transition_kernel, 0.5)
    tk_low7, tk_up7 = Interval.compute_interval(forest.transition_kernel, 0.01)
    tk_low8, tk_up8 = Interval.compute_interval(forest.transition_kernel, 1.0)
    tk_low9, tk_up9 = Interval.compute_interval(forest.transition_kernel, 3.0)

    # tk_low = (tk-0.5).clip(min=0)
    # tk_up = (tk+0.5).clip(max=1)

    # problems can also be supplied as a list
    evaluator = Evaluator(
        forest,
        [
            Robust(Wasserstein(0.01)),
            Robust(Wasserstein(0.05)),
            Robust(Wasserstein(0.16)),
            Robust(Wasserstein(0.1)),
            Robust(Wasserstein(0.2)),
            Robust(Wasserstein(0.3)),
            Robust(Wasserstein(0.4)),
            Robust(Wasserstein(0.5)),
            Robust(Wasserstein(0.6)),
            Robust(Wasserstein(0.7)),
            Robust(Wasserstein(1.1)),
            Robust(Wasserstein(2.0)),
            Robust(Ellipsoid(0.01)),
            Robust(Ellipsoid(0.098)),
            Robust(Ellipsoid(0.12)),
            Robust(Ellipsoid(0.06)),
            Robust(Ellipsoid(0.22)),
            Robust(Ellipsoid(0.32)),
            Robust(Ellipsoid(0.72)),
            Robust(Ellipsoid(1.22)),
            ValueIteration,
            Robust(Likelihood(4.1, 0.001)),
            Robust(Likelihood(5.1, 0.001)),
            Robust(Likelihood(3.1, 0.001)),
            Robust(Likelihood(2.1, 0.001)),
            Robust(Likelihood(1.1, 0.001)),
            Robust(Likelihood(6.1, 0.001)),
            Robust(Likelihood(7.1, 0.001)),

            #range 1.5 - 0ish
            Robust(Interval(tk_low1, tk_up1)),
            Robust(Interval(tk_low2, tk_up2)),
            Robust(Interval(tk_low3, tk_up3)),
            Robust(Interval(tk_low4, tk_up4)),
            Robust(Interval(tk_low5, tk_up5)),
            Robust(Interval(tk_low6, tk_up6)),
            Robust(Interval(tk_low7, tk_up7)),
            Robust(Interval(tk_low8, tk_up8)),
            Robust(Interval(tk_low9, tk_up9))
        ],
        Options(
            number_of_paths=1000, # set size of in and out set
            number_of_sims=1000,
            plot_hist=True,
            do_simulation=False,
            evaluate_all=True,
            evaluate_inner=False,
            evaluate_outer=False,
            sample_var=0.5,
            sample_amount=3000, # is used for all_samples
            sample_uniform=False,
            non_robust_actions=[], # replace with [1] if for action there should be no robustness
            variance_scaling=True,
            variance_lower=0,
            variance_upper=1,
            logging_behavior=LoggingBehavior.TABLE
        ))
    evaluator.run()


run_default()
