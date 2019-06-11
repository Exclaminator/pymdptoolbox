import Evaluator

"""
default configuration, runs the forest problem on some models
"""


def run_default():
    options_dict = {
        "number_of_paths": 100
    }
    problem_dict = {
        "format": "list",
        "elements": [
            {
                "type": "forest",
                "parameters": {
                    "variance": 0.1
                }
            }
        ]
    }
    mdp_dict = {
        "elements": [
            {
                "type": "robust",
                "parameters": {
                    "sigma_identifier": "ellipsoid"
                },
            },
            {
                "type": "robust",
                "parameters": {
                    "sigma_identifier": "wasserstein"
                },
            },
            {
                "type": "robust",
                "parameters": {
                    "sigma_identifier": "interval"
                },
            },
            {
                "type": "valueIteration",
                "parameters": {}
            }
        ]
    }
    evaluator = Evaluator(problem_dict, options_dict)
    evaluator.run(mdp_dict)
