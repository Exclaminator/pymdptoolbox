from Evaluator import Evaluator
from Options import Options
import model_picker as mp

"""
default configuration, runs the forest problem on some models
"""


def run_default():
    options_dict = {
        Options.NUMBER_OF_PATHS: 100,
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
            {
                Options.TYPE_KEY: "robust",
                Options.PARAMETERS_KEY: {
                    mp.SIGMA_IDENTIFIER_KEY: mp.ELLIPSOID_KEY
                },
            },
            # {
            #     Options.TYPE_KEY: mp.ROBUST_KEY,
            #     Options.PARAMETERS_KEY: {
            #         mp.SIGMA_IDENTIFIER_KEY: mp.MAX_LIKELIHOOD_KEY,
            #         "delta": 0.1
            #     },
            # },
            # {
            #     "type": "robust",
            #     "parameters": {
            #         "sigma_identifier": "wasserstein"
            #     },
            # },
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
