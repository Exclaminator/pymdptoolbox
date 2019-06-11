import Options
import mdp_base
import mdptoolbox.Robust

"""
Module to help creating a corresponding MDP object.
"""

RANDOM_MDP_KEY = "random"
ROBUST_MDP_KEY = "robust"
VALUE_MDP_KEY = "valueIteration"

SIGMA_IDENTIFIER_KEY = "sigma_identifier"
ELLIPSOID_KEY = "ellipsoid"
WASSERSTEIN_KEY = "wasserstein"
INTERVAL_KEY = "interval"
MAX_LIKELIHOOD_KEY = "max_likelihood"

def create_mdp(mdp_as_dict, problem):
    # todo: cleanup and make it nice
    mdp_type = mdp_as_dict[Options.TYPE_KEY]
    mdp_hyperparameters = mdp_as_dict[Options.PARAMETERS_KEY]

    transition_kernel = problem.transition_kernel
    reward_matrix = problem.reward_matrix

    # discount factor is problem specific
    discount_factor = problem.discount_factor

    mdp_out = None
    # define mdp_out based on the type and any hyperparameters

    if mdp_type == RANDOM_MDP_KEY:
        mdp_out = mdp_base.RandomMdp(transition_kernel.ttk, reward_matrix, None, None, None, None)

    elif mdp_type == ROBUST_MDP_KEY:
        #####
        # todo: let this blend with youri's changes for the robust MDP model

        sigma_identifier = mdp_as_dict[SIGMA_IDENTIFIER_KEY]


        mdp_out = mdptoolbox.Robust.RobustModel(
            transition_kernel, reward_matrix, discount=discount_factor,
            p_lower=interval["p_low"], p_upper=interval["p_up"],
            sigma_identifier=retrieve_from_dict(mdp_hyperparameters, "sigma_identifier", "interval")
        )
        if mdp_hyperparameters["sigma_identifier"] == "ellipsoid":
            mdp_out.max_iter = 10
        #####

    elif mdp_type == VALUE_MDP_KEY:
        mdp_out = mdptoolbox.mdp.ValueIteration(transition_kernel.ttk, reward_matrix, discount=discount_factor)
        mdp_out.max_iter = 10000
    mdp_out.run()
    return mdp_out

def _get_sigma_function(identifier):
    """
    Returns a corresponding sigma function, which can be used as an input for the robust methods.
    """
    # todo: implement based on youri's implementation
    if identifier == INTERVAL_KEY:



    return