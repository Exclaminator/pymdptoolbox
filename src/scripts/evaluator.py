import sys
import mdp_base
import numpy as np
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

    results_all = []
    # for each MDP
    for mdp_pair in mdp_pair_list:
        result_mdp = []
        # run on all problems
        for problem in problem_list:
            result_problem = []
            # instantiate mdp
            mdp = create_mdp_from_pair(mdp_pair, problem["P"], problem["R"])
            # simulate some number of time
            for ii in range(number_of_runs):
                result_problem.append(
                    run_policy_on_problem(
                        mdp.policy, problem["P"],
                        problem["R"],
                        problem["t_max"]
                    )
                )
            result_mdp.append(result_problem)

        # do evaluation on results for this mdp and log it
        evaluated_results = evaluate_mdp_results(result_mdp, options)
        write_to_log(evaluated_results, log_file)

        results_all.append(result_mdp)

    # If we want to do some evaluation over the total set of results, we should do that here

    return results_all


"""
policy: policy retrieved by mdp
t_max: maximum amount of state transitions to consider
P: transition kernel
R: reward kernel
"""


def run_policy_on_problem(policy, t_max, P, R):
    s = 0
    total_reward = 0

    for t in range(t_max):
        action = policy[s]
        PP = P[action, s]
        s_new = np.random.choice(a=len(PP), p=PP)
        RR = R[s]
        total_reward += RR[action]
        s = s_new

    return total_reward


def create_problem_from_environment_description(options, environment):
    # todo: based on the options and environments,
    #  create a list of problems that indicate what problems are used for evaluation

    # output should be a list of (P, R) pairs, where P is the transition kernel and R the reward kernel
    return [{"P": -1, "R": -1, "t_max": -1}, {"P": -2, "R": -2, "t_max": -2}]


"""
Function that creates the corresponding mdp given a (mdptype, parameters) pair
"""


def create_mdp_from_pair(pair, problem):
    mdp_type = pair[0]
    mdp_parameters = pair[1]

    # todo: create mdp object given its type and parameters
    # look at https://stackoverflow.com/questions/452969/does-python-have-an-equivalent-to-java-class-forname
    return mdp_base.RandomMdp(problem["P"], problem["R"], None, None, None, None)


def create_options_object(options_raw):
    # todo: create an options object that is easy to handle in python
    # I think we should use dict for this
    return {}


def evaluate_mdp_results(result_mdp, options):
    # todo: take the results and make some sens out of it
    # what we want to retrieve should be defined in the options
    return {}


def write_to_log(content, filename):
    # todo: implement this method

"""
Main code to run, which takes arguments and calls functions
"""
# first argument is script filename, so we skip that one
args = sys.argv
mdp_pair_list = args[1]
number_of_runs = args[2]
options = args[3]
log_file = args[4]
environment_description = args[5]

run_multi(mdp_pair_list, number_of_runs, options, log_file)
