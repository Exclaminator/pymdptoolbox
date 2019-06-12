import numpy as _np
import mdptoolbox.example
from TransitionKernel import TransitionKernelVar, TransitionKernelInterval
import abc
from Options import Options


"""
creates a problem from a dict.
Selects the corresponding type
"""


def create_problem_by_dict(problem_dict):
    problem_type = problem_dict[Options.TYPE_KEY]
    parameters_dict = problem_dict[Options.PARAMETERS_KEY]
    if problem_type == ForestProblem.KEY:
        return ForestProblem.from_dict(parameters_dict)
    elif problem_type == RandomProblem.KEY:
        return RandomProblem.from_dict(parameters_dict)


def _retrieve_else(dictionary, field, default):
    return dictionary[field] if field in dictionary else default


class Problem(object):

    KEY = Options.DEFAULT_KEY
    discount_KEY = "discount_factor"
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

    def get_key(self):
        return self.KEY


class ForestProblem(Problem):

    KEY = "forest"

    S_KEY = "S"
    r1_KEY = "r1"
    r2_KEY = "r2"
    p_KEY = "p"
    p_low_KEY = "p_low"
    p_up_KEY = "p_up"

    S_def = 10
    r1_def = 40
    r2_def = 1
    p_def = 0.05
    p_low_def = 0.01
    p_up_def = 0.10

    """
    adds uncertainty to the probability of fire p
    """
    def __init__(self, S, discount_factor, r1, r2, p, p_low, p_up):
        ttk, reward_matrix = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)

        # recompute ttk for upper and lower value of p
        ttk_low, reward_low = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p_low)
        ttk_up, reward_up = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p_up)

        # S x A -> A x S x S'
        reward_matrix = _np.transpose(reward_matrix)
        reward_matrix = _np.repeat(reward_matrix[:, :, _np.newaxis], reward_matrix.shape[1], axis=2)

        # create problem
        Problem.__init__(self, TransitionKernelInterval(ttk, ttk_low, ttk_up), reward_matrix, discount_factor)

    @staticmethod
    def from_dict(parameters_dict):
        return ForestProblem(
            S=_retrieve_else(parameters_dict, ForestProblem.S_KEY, ForestProblem.S_def),
            discount_factor=_retrieve_else(parameters_dict, Problem.discount_KEY, Problem.discount_def),
            r1=_retrieve_else(parameters_dict, ForestProblem.r1_KEY, ForestProblem.r1_def),
            r2=_retrieve_else(parameters_dict, ForestProblem.r2_KEY, ForestProblem.r2_def),
            p=_retrieve_else(parameters_dict, ForestProblem.p_KEY, ForestProblem.p_def),
            p_low=_retrieve_else(parameters_dict, ForestProblem.p_low_KEY, ForestProblem.p_low_def),
            p_up=_retrieve_else(parameters_dict, ForestProblem.p_up_KEY, ForestProblem.p_up_def)
        )


class RandomProblem(Problem):

    KEY = "random_highest_p"
    S_KEY = "S"
    A_KEY = "A"
    var_KEY = "var"
    only_highest_p_KEY = "only_highest"

    S_def = 10
    A_def = 5
    var_def = "var"
    only_highest_p_def = True

    def __init__(self, S, A, variance, only_highest_p=True):
        ttk, reward_matrix = mdptoolbox.example.rand(S, A, is_sparse=False)
        reward_matrix = _np.maximum(reward_matrix, 0)

        # get most probable transition probabilities per state-action
        ttk_argmax = _np.argmax(ttk, 2)
        ttk_var = _np.zeros(ttk.shape)

        for action_index in range(A):
            for state_index in range(S):
                if only_highest_p:
                    ttk_var[action_index, state_index, ttk_argmax[action_index, state_index]] = variance
                else:
                    # todo: test this, but this probably works
                    ttk_var[action_index, state_index, :] = variance

        Problem.__init__(self, TransitionKernelVar(ttk, ttk_var), reward_matrix)

    @staticmethod
    def from_dict(parameters_dict):
        return RandomProblem(
            S=_retrieve_else(parameters_dict, RandomProblem.S_KEY, RandomProblem.S_def),
            A=_retrieve_else(parameters_dict, RandomProblem.A_KEY, RandomProblem.A_def),
            variance=_retrieve_else(parameters_dict, RandomProblem.var_KEY, RandomProblem.var_def),
            only_highest_p=_retrieve_else(
                parameters_dict, RandomProblem.only_highest_p_KEY, RandomProblem.only_highest_p_def
            )
        )
