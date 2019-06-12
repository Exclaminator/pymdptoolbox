from Evaluator import Evaluator
from Options import Options

"""
default configuration, runs the forest problem on some models
"""


def run_default():
    options_dict = {
        "number_of_paths": 100,
        Options.PLOT_HIST: True,
        Options.DO_SIMULATION: True
    }
    problem_dict = {
        "format": "list",
        "elements": [
            {
                "type": "forest",
                "parameters": {
                    "p_low": 0.01,
                    "p_up": 0.5
                }
            }
        ]
    }
    mdp_dict = {
        "elements": [
            # {
            #     "type": "robust",
            #     "parameters": {
            #         "sigma_identifier": "ellipsoid"
            #     },
            # },
            {
                "type": "robust",
                "parameters": {
                    "sigma_identifier": "wasserstein"
                },
            },
            # {
            #     "type": "robust",
            #     "parameters": {
            #         "sigma_identifier": "interval"
            #     },
            # },
            {
                "type": "valueIteration",
                "parameters": {}
            }
        ]
    }
    evaluator = Evaluator(problem_dict, options_dict)
    evaluator.run(mdp_dict)


run_default()
