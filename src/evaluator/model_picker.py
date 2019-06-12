from Options import Options
from mdptoolbox.Robust import RobustModel
import mdptoolbox.mdp

"""
Module to help creating a corresponding MDP object.
"""

ROBUST_KEY = "robust"
VALUE_KEY = "valueIteration"

SIGMA_IDENTIFIER_KEY = "sigma_identifier"

INTERVAL_KEY = "interval"
P_LOW_KEY = "p_low"
P_UP_KEY = "p_up"

ELLIPSOID_KEY = "ellipsoid"
BETA_KEY = "beta"
WASSERSTEIN_KEY = "wasserstein"
MAX_LIKELIHOOD_KEY = "max_likelihood"
DELTA_KEY = "delta"


def create_mdp(mdp_as_dict, problem):
    mdp_type = mdp_as_dict[Options.TYPE_KEY]
    mdp_hyperparameters = mdp_as_dict[Options.PARAMETERS_KEY]

    transition_kernel = problem.transition_kernel
    reward_matrix = problem.reward_matrix

    # discount factor is problem specific
    discount_factor = problem.discount_factor

    mdp_out = None
    # define mdp_out based on the type and any hyperparameters

    if mdp_type == ROBUST_KEY:
        # todo: test this
        sigma_function = _get_sigma_function(mdp_hyperparameters[SIGMA_IDENTIFIER_KEY],
                                             transition_kernel, mdp_hyperparameters)
        mdp_out = RobustModel(
            transition_kernel.ttk, reward_matrix, discount=discount_factor, innerfunction=sigma_function
        )

    elif mdp_type == VALUE_KEY:
        mdp_out = mdptoolbox.mdp.ValueIteration(transition_kernel.ttk, reward_matrix, discount=discount_factor)
        mdp_out.max_iter = 10000
    mdp_out.run()
    return mdp_out


def _get_sigma_function(identifier, transition_kernel, mdp_hyperparameters):
    """
    Returns a corresponding sigma function, which can be used as an input for the robust methods.
    """
    # todo: implement based on youri's implementation

    # todo: at the moment the robust models assume the ambiguity is of a certain type, to make sure the intervals match

    if identifier == INTERVAL_KEY:
        return RobustModel.innerMethod.Interval(transition_kernel.ttk_low, transition_kernel.ttk_up)
    elif identifier == ELLIPSOID_KEY:
        return RobustModel.innerMethod.Elipsoid(transition_kernel.beta)
    elif identifier == WASSERSTEIN_KEY:
        return RobustModel.innerMethod.Wasserstein(transition_kernel.beta)
    elif identifier == MAX_LIKELIHOOD_KEY:
        return RobustModel.innerMethod.Likelihood(transition_kernel.beta, mdp_hyperparameters[DELTA_KEY])
    else:
        raise ValueError("invalid sigma identifier: " + identifier)
