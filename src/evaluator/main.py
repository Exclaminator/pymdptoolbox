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
    tk = forest.transition_kernel
    tk_low = (tk-0.5).clip(min=0)
    tk_up = (tk+0.5).clip(max=1)

    # problems can also be supplied as a list
    evaluator = Evaluator(
        forest,
        [
            Robust(Wasserstein(0.7)),
            Robust(Ellipsoid(0.01)),
            ValueIteration,
            Robust(Likelihood(1.4, 0.001)), #range 1.5 - 0
            Robust(Interval(tk_low, tk_up))
        ],
        Options(
            number_of_paths=100,
            number_of_sims=1000,
            plot_hist=True,
            do_simulation=False,
            evaluate_all=True,
            evaluate_inner=True,
            sample_var=0.15,
            sample_amount=1000,
            sample_uniform=False
        ))
    evaluator.run()


run_default()
