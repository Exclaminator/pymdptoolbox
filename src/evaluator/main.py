from Evaluator import Evaluator
from Options import Options
from mdptoolbox.Robust import *
from mdptoolbox.mdp import ValueIteration

from Problem import Problem


"""
default configuration, runs the forest problem on some models
"""


def run_default():
    # get tk_low and tk_up for the interval model
    forest = Problem.get_forest_problem(S=10, discount_factor=0.9, r1=10, r2=2, p=0.05)
    tk_low, tk_up = Interval.compute_interval(forest.transition_kernel, 0.0138)
    # tk_low = (tk-0.5).clip(min=0)
    # tk_up = (tk+0.5).clip(max=1)

    # problems can also be supplied as a list
    evaluator = Evaluator(
        forest,
        [
            Robust(Wasserstein(0.07)),
            Robust(Ellipsoid(0.13)),
            ValueIteration,
            Robust(Likelihood(0.6215, 0.001)), #range 1.5 - 0
            Robust(Interval(tk_low, tk_up))
        ],
        Options(
            number_of_paths=100,
            number_of_sims=1000,
            plot_hist=True,
            do_simulation=False,
            evaluate_all=True,
            evaluate_inner=True,
            evaluate_outer=True,
            sample_var=0.05,
            sample_amount=10000,
            sample_uniform=False
        ))
    evaluator.run()


run_default()
