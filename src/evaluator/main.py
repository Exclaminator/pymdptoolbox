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

    forestProblem = Problem.create_forest_problem()
    tk = forestProblem.transition_kernel
    tk_low = normalize_tk(tk-0.01)
    tk_up = normalize_tk(tk+0.05)

    options = Options(
        number_of_paths=500,
        number_of_runs=100,
        plot_hist=True,
        do_simulation=True,
        evaluate_all=False,
        sample_var=0.15,
        sample_amount=1000
    )
    problem_dict = {
        "forest": forestProblem
    }
    mdp_dict = {
        "wasserstein-0.1": Robust(Wasserstein(0.1)),
        # "ellipsoid-0.1": Robust(Ellipsoid(0.1)),
        "value_iteration": ValueIteration,
        "max_likelihood-0.1-0.1": Robust(Likelihood(0.1, 0.1)),
        # "interval": Robust(Interval(tk_low, tk_up))
    }
    Evaluator.build_and_run(problem_dict, mdp_dict, options)


run_default()
