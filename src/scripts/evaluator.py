import sys
import os
from datetime import datetime
import mdp_base
import mdptoolbox.example
import mdptoolbox.Robust
import numpy as _np
from matplotlib import pyplot
import seaborn as sns

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


def run_multi(mdp_pair_list, number_of_runs, options, problems_dict):

    # define problems to run on
    problem_list = create_problem_list(options, problems_dict)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    # retrieve variables from options file
    log_filename = retrieve_from_dict(dictionary=options, field="log_filename", default="../../logs/" + timestamp + ".log")
    figure_path = retrieve_from_dict(options, "figure_path", "../../logs/" + timestamp + "_fig/")
    plot_disabled = retrieve_from_dict(options, "plot_disabled", False)

    if ~plot_disabled:
        os.mkdir(figure_path)

    # create log file
    file_to_write = open(log_filename, "w+")

    results_all = []
    # for each MDP
    for mdp_pair in mdp_pair_list:
        file_to_write.write("\n"+str(mdp_pair)+"\n")

        result_mdp = []
        # run on all problems
        for problem in problem_list:
            result_problem = []
            # instantiate mdp
            mdp = create_mdp_from_dict(mdp_pair, problem, options)
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

            # save a plot to the figure folder
            if ~plot_disabled:
                description = problem["problem_name"]+"_"+mdp_pair["type"]

                sns.distplot(result_problem)
                pyplot.title(description)
                pyplot.xlabel("Value")
                pyplot.ylabel("Frequency")
                pyplot.savefig(figure_path+description+".png", dpi=150, format="png")
                pyplot.close()

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


def compute_interval_by_variance(P, P_var):
    # we do so by assuming the variance corresponds to an uniform distribution.
    # Then we compute p_up (b) and p_low (a), the lower and upper bound, using the formulations for variance and mean
    # var(X) = (b - a)^2 / 12
    # mu(X) = (a+b) / 2
    # doing some algebra gives us
    # b = mu + \sqrt(3*var)
    # a = mu - \sqrt(3*var)
    sqrt3var = _np.sqrt(3*P_var)

    return {"p_up": P + sqrt3var, "p_low": P - sqrt3var}


def create_problem_list(options_object, problems_dict):
    # create a list of problems that indicate what problems are used for evaluation.
    # output should be a list of problems.
    # P is the transition kernel, R the reward kernel,
    # t_max how many steps are taken, P_var the variance (uncertainty) of the true transition kernel.

    environment_format = problems_dict["format"]
    result = []

    if environment_format == "list":
        for problem in problems_dict["list"]:
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


def create_mdp_from_dict(mdp_as_dict, problem, options):
    mdp_type = mdp_as_dict["type"]
    mdp_hyperparameters = mdp_as_dict["parameters"]
    P = problem["P"]
    R = problem["R"]

    # not quite sure whether we should include discount factor as part of the problem, or as part of the mdp
    # I chose "as part of the mdp" for now
    discount_factor = retrieve_from_dict(mdp_hyperparameters, "discount_factor", 0.9)

    mdp_out = None
    # define mdp_out based on the type and any hyperparameters
    if mdp_type == "randomMdp":
        mdp_out = mdp_base.RandomMdp(P, R, None, None, None, None)
    elif mdp_type == "robustInterval":
        interval = compute_interval_by_variance(P, problem["P_var"])
        mdp_out = mdptoolbox.Robust.RobustIntervalModel(
            P, R, discount=discount_factor,
            p_lower=interval["p_low"], p_upper=interval["p_up"])
    elif mdp_type == "valueIteration":
        mdp_out = mdptoolbox.mdp.ValueIteration(P, R, discount=discount_factor)

    mdp_out.run()
    return mdp_out


def evaluate_mdp_results(result_mdp, options):
    # ake the results + options object and define how we want to log

    logging_behavior = retrieve_from_dict(options, "logging_behavior", "default")

    average_reward = _np.mean(result_mdp)
    variance = _np.var(result_mdp)
    lowest_value = _np.min(result_mdp)

    # define logging behavior here
    # maybe add "verbose" or "minimal" etc.
    if logging_behavior == "default":
        return {
            "average_value": average_reward,
            "variance": variance,
            "lowest_value": lowest_value
        }
    return {}


def retrieve_from_dict(dictionary, field, default):
    return dictionary[field] if field in dictionary else default


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
        },
        {
            "type": "valueIteration",
            "parameters": {}
        }
    ],
    number_of_runs=100,
    options={
        "t_max_def": 100,
        "save_figures": True,
        "logging_behavior": "Default",
        # "log_filename": "dsadsa"
        #"figure_save_path": "../../figures"
        "plot_disabled": False
    },
    problems_dict={
        "format": "list",
        "list": ["forest_default", "forest_risky"]
    }
)

