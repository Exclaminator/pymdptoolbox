import Evaluator
from Options import Options
from mdptoolbox.Robust import Robust
from ProblemSet import normalize_tk

from mdptoolbox.InnerMethod.Wasserstein import Wasserstein
from mdptoolbox.InnerMethod.Ellipsoid import Ellipsoid
from mdptoolbox.InnerMethod.Likelihood import Likelihood
from mdptoolbox.InnerMethod.Interval import Interval

from mdptoolbox.mdp import ValueIteration

from Problem import Problem


"""
default configuration, runs the forest problem on some models
"""


def run_default():

    # get tk_low and tk_up for the interval model
    forestProblem = Problem.create_forest_problem(S=10, discount_factor=0.9, r1=10, r2=2, p=0.05)
    tk = forestProblem.transition_kernel
    tk_low = (tk-0.7).clip(min=0)
    tk_up = (tk+0.4).clip(max=1)

    options = Options(
        number_of_paths=100,
        number_of_sims=1000,
        plot_hist=True,
        do_simulation=False,
        evaluate_all=True,
        evaluate_inner=True,
        sample_var=0.1,
        sample_amount=1000,
    )
    problem_dict = {
        #"forest": forestProblem,
        "random_problem": Problem.create_random_problem()
    }
    mdp_dict = {
        "wasserstein-0.055": Robust(Wasserstein(0.3)),
        "ellipsoid-0.3": Robust(Ellipsoid(5)),
        "value_iteration": ValueIteration,
        #"max_likelihood-0.2-0.2": Robust(Likelihood(0.01, 0.2)),
        #"interval": Robust(Interval(tk_low, tk_up))
    }
    Evaluator.build_and_run(problem_dict, mdp_dict, options)


run_default()
