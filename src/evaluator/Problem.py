import numpy as _np
import mdptoolbox.example
from TransitionKernel import TransitionKernelVar, TransitionKernelInterval
import abc
from Options import Options


"""
creates a problem from a dict.
Selects the corresponding type
"""


# def create_problem_by_dict(problem_dict):
#     problem_type = problem_dict[Options.TYPE_KEY]
#     parameters_dict = problem_dict[Options.PARAMETERS_KEY]
#     if problem_type == ForestProblem.KEY:
#         return ForestProblem.from_dict(parameters_dict)
#     elif problem_type == RandomProblem.KEY:
#         return RandomProblem.from_dict(parameters_dict)


# def _retrieve_else(dictionary, field, default):
#     return dictionary[field] if field in dictionary else default


class Problem(object):

    # KEY = Options.DEFAULT_KEY
    # discount_KEY = "discount_factor"
    discount_def = 0.9

    """
    Creates a problem object. Uses the example module.
    We can initialize it based on some parameters.
    Next, we can extract P, P_var and R from it, which can be put into a MDP object.

    Types of ambiguities
    - interval based (p_low <= p <= p_up)
    - variance based (p = P +/- sqrt(p_var))
    - distance based (d(p,P) <= beta)
    """

    def __init__(self, transition_kernel, reward_matrix, discount_factor):
        self.transition_kernel = transition_kernel
        self.reward_matrix = reward_matrix
        self.discount_factor = discount_factor

    @staticmethod
    def create_forest_problem(state_amount=10, discount_factor=discount_def, r1=40, r2=20, p=0.05):
        tk, reward_matrix = mdptoolbox.example.forest(S=state_amount, r1=r1, r2=r2, p=p)

        # S x A -> A x S x S'
        reward_matrix = _np.transpose(reward_matrix)
        reward_matrix = _np.repeat(reward_matrix[:, :, _np.newaxis], reward_matrix.shape[1], axis=2)

        # create problem
        return Problem(tk, reward_matrix, discount_factor)

    @staticmethod
    def create_random_problem(state_amount=10, A=5, discount_factor=discount_def):
        tk, reward_matrix = mdptoolbox.example.rand(state_amount, A, is_sparse=False)
        reward_matrix = _np.maximum(reward_matrix, 0)

        return Problem(tk, reward_matrix, discount_factor)

