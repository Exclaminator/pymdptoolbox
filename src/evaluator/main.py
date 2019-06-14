import Evaluator
from Options import Options
from mdptoolbox.Robust import Robust

from mdptoolbox.InnerMethod.Wasserstein import Wasserstein
from mdptoolbox.InnerMethod.Ellipsoid import Ellipsoid
from mdptoolbox.mdp import ValueIteration

from Problem import Problem


"""
default configuration, runs the forest problem on some models
"""


def run_default():
    options = Options(
        number_of_paths=100,
        number_of_runs=100,
        plot_hist=True,
        do_simulation=True,
        evaluate_all=False,
        sample_var=0.15,
        sample_amount=1000
    )
    problem_dict = {
        "forest": Problem.create_forest_problem()
    }
    mdp_dict = {
        "wasserstein-0.1": Robust(Wasserstein(0.1)),
        #"ellipsoid-0.1": Robust(Ellipsoid(0.1)),
        "value_iteration": ValueIteration
    }
    Evaluator.build_and_run(problem_dict, mdp_dict, options)


run_default()

# def run_default_old():
#     options_dict = {
#         Options.NUMBER_OF_PATHS: 100,
#         Options.PLOT_HIST: True,
#         Options.DO_SIMULATION: True
#     }
#     problem_dict = {
#         "format": "list",
#         "elements": [
#             {
#                 "type": "forest",
#                 "parameters": {
#                     "p_low": 0.01,
#                     "p_up": 0.5
#                 }
#             }
#         ]
#     }
#     mdp_dict = {
#         "elements": [
#             {
#                 Options.TYPE_KEY: "robust",
#                 Options.PARAMETERS_KEY: {
#                     mp.SIGMA_IDENTIFIER_KEY: mp.ELLIPSOID_KEY
#                 },
#             },
#             {
#                 Options.TYPE_KEY: mp.ROBUST_KEY,
#                 Options.PARAMETERS_KEY: {
#                     mp.SIGMA_IDENTIFIER_KEY: mp.MAX_LIKELIHOOD_KEY,
#                     "delta": 0.1,
#                     "beta": 0.1
#                 },
#             },
#             {
#                 "type": "robust",
#                 "parameters": {
#                     "sigma_identifier": "wasserstein",
#                     "beta": 0.1
#                 },
#             },
#             # {
#             #     "type": "robust",
#             #     "parameters": {
#             #         "sigma_identifier": "interval"
#             #     },
#             # },
#             {
#                 "type": "valueIteration",
#                 "parameters": {}
#             }
#         ]
#     }
#     evaluator = Evaluator(problem_dict, options_dict)
#     evaluator.run(mdp_dict)


