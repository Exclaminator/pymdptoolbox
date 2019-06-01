import sys
from datetime import datetime
import mdp_base
import mdptoolbox.example
import mdptoolbox.Robust
import numpy as _np
"""
Version 1:

We don't consider tweaking parameters.

Input:
- list of mdp-type's and their corresponding hyperparameters.
- An integer, indicating how many runs we want to do.
- Options: list of keywords that indicate what metrics are outputted.
- log_file: where to write our found results to. 
    Default: timestamp + current folder
- Environment: problem we want to test on, given by some keyword. 
    Default: behavior will be based on the options given, i.e. what we want to measure

Output:
- depends on 'Options'

"""


def run_multi(mdp_pair_list, number_of_runs, options, log_file, environment):

    # define problems to run on
    problem_list = create_problem_from_environment_description(options, environment)

    # create log file
    file_to_write = open(log_file, "w+")

    results_all = []
    # for each MDP
    for mdp_pair in mdp_pair_list:
        file_to_write.write("\n"+str(mdp_pair)+"\n")

        result_mdp = []
        # run on all problems
        for problem in problem_list:
            result_problem = []
            # instantiate mdp
            mdp = create_mdp_from_dict(mdp_pair, problem)
            # simulate some number of time
            for ii in range(number_of_runs):
                result_problem.append(
                    run_policy_on_problem(
                        mdp.policy, problem
                    )
                )
            # do evaluation on results for this mdp and log it
            file_to_write.write(problem["problem_name"]+":\n")
            file_to_write.write("policy: "+str(mdp.policy)+"\n")
            file_to_write.write(str(evaluate_mdp_results(result_problem, options))+"\n")

            result_mdp.append(result_problem)

        results_all.append(result_mdp)

    # If we want to do some evaluation over the total set of results, we should do that here

    file_to_write.close()
    return results_all


"""
policy: policy retrieved by mdp
t_max: maximum amount of state transitions to consider
P: transition kernel
R: reward kernel
"""


def run_policy_on_problem(policy, problem):
    s = 0
    total_reward = 0

    P = problem["P"]
    P_var = problem["P_var"]
    P_std = _np.sqrt(P_var)
    t_max = problem["t_max"]
    R = problem["R"]

    for t in range(t_max):
        action = policy[s]

        # simulate ambiguity: simulate a transition probability based on the variance
        # we use a normal distribution for now, but we might want to consider other distributions
        probs = _np.random.normal(P[action, s], P_std[action, s])
        # we can't have a negative probabilities, so we take the absolute value
        PP = _np.absolute(probs, _np.zeros(P[action, s].shape))

        # normalize to make sum = 1
        PP2 = PP/sum(PP)

        s_new = _np.random.choice(a=len(PP2), p=PP2)
        RR = R[s]
        total_reward += RR[action]
        s = s_new

    return total_reward


def computeIntervalByVariance(P, P_var):
    # we do so by assuming the variance corresponds to an uniform distribution.
    # Then we compute p_up (b) and p_low (a), the lower and upper bound, using the formulations for variance and mean
    # var(X) = (b - a)^2 / 12
    # mu(X) = (a+b) / 2
    # todo: compute a and b given mu and var, i.e. replace P with a and b
    return {"p_up": P, "p_low": P}


def create_problem_from_environment_description(options_object, environment):
    # todo: based on the options and environments,
    #  create a list of problems that indicate what problems are used for evaluation

    # output should be a list of problems.
    # P is the transition kernel, R the reward kernel,
    # t_max how many steps are taken, P_var the variance (uncertainty) of the true transition kernel

    environment_format = environment["format"]
    result = []

    if environment_format == "list":
        for problem in environment["problem_list"]:
            problem_to_add = {}
            if problem == "forest_default":
                P, R = mdptoolbox.example.forest()
                P_var = _np.full(P.shape, 0.5)
                problem_to_add = {"P": P, "R": R, "P_var": P_var, "t_max": options_object["t_max_def"]}

            elif problem == "forest_risky":
                S = 10  # number of states
                r1 = 40  # reward when 'wait' is performed in its oldest state
                r2 = 1  # reward when 'cut' is performed in its oldest state
                p = 0.05  # probability of fire
                P, R = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)
                P_var = _np.full(P.shape, 1)
                problem_to_add = {"P": P, "R": R, "P_var": P_var, "t_max": options_object["t_max_def"]}

            problem_to_add["problem_name"] = problem
            result.append(problem_to_add)

    return result


"""
Function that creates the corresponding mdp
"""


def create_mdp_from_dict(mdp_as_dict, problem):
    mdp_type = mdp_as_dict["type"]
    mdp_parameters = mdp_as_dict["parameters"]

    mdp_out = None
    # todo: create mdp object given its type and parameters
    if mdp_type == "randomMdp":
        mdp_out = mdp_base.RandomMdp(problem["P"], problem["R"], None, None, None, None)
    elif mdp_type == "robustInterval":
        interval = computeIntervalByVariance(problem["P"], problem["P_var"])
        mdp_out = mdptoolbox.Robust.RobustIntervalModel(
            problem["P"], problem["R"], discount=mdp_parameters["discount_factor"],
            p_lower=interval["p_low"], p_upper=interval["p_up"])

    mdp_out.run()
    return mdp_out


# def create_options_object(options_raw):
#     # todo: create an options object that is easy to handle in python
#     # could have, for now we can just pass a dict object in run_multi
#     return {}


def evaluate_mdp_results(result_mdp, options):
    # todo: take the results and make some sens out of it
    # what we want to retrieve should be defined in the options

    average_reward = _np.mean(result_mdp)
    variance = _np.var(result_mdp)
    lowest_value = _np.min(result_mdp)

    return {
        "average_value": average_reward,
        "variance": variance,
        "lowest_value": lowest_value
    }


"""
Main code to run, which takes arguments and calls functions
"""
# first argument is script filename, so we skip that one
# args = sys.argv
# mdp_pair_list = args[1]
# number_of_runs = args[2]
# options = args[3]
# log_file = args[4]
# environment_description = args[5]

# run_multi(mdp_pair_list, number_of_runs, options, log_file)

# test with some default parameters
run_multi(
    mdp_pair_list=[
        {
            "type": "robustInterval",
            "parameters": {
                "discount_factor": 0.9
            }
        },
        {
            "type": "randomMdp",
            "parameters": {}
        }
    ],
    number_of_runs=10,
    options={
        "t_max_def": 100
    },
    log_file="../../logs/"+datetime.now().strftime('%Y%m%d-%H%M%S')+".log",
    environment={
        "format": "list",
        "problem_list": ["forest_default", "forest_risky"]
    }
)

