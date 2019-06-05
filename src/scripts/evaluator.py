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
MDP-multi evaluation tool.
Creates a list of mdp's and problems. Then runs each mdp-problem combination a fixed number of times.
Can save figures and log results
"""


def run_multi(mdp_pair_list, number_of_runs, options, problems_dict):

    # define problems to run on
    problem_list = create_problem_list(options, problems_dict)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    # if not exists create folder
    log_folder = "../../logs/"

    # retrieve variables from options file
    log_filename = retrieve_from_dict(dictionary=options, field="log_filename", default=log_folder + timestamp + ".log")
    figure_path = retrieve_from_dict(options, "figure_path", log_folder + timestamp + "_fig/")
    plot_disabled = retrieve_from_dict(options, "plot_disabled", False)

    os.makedirs(figure_path)

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
                description = problem["problem_name"]+"-"+mdp_pair["type"]

                sns.distplot(result_problem)
                pyplot.title(description)
                pyplot.xlabel("Value")
                pyplot.ylabel("Frequency")
                pyplot.savefig(figure_path+description+".png", dpi=150, format="png")
                pyplot.close()

            result_mdp.append(result_problem)

        results_all.append(result_mdp)

    # If we want to do some evaluation over the total set of results we can do that here

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
        # R is in format A x S x S'
        RR = R[:, s, s_new]
        total_reward += RR[action]
        s = s_new

    return total_reward


def compute_interval_by_variance(P, P_var, z=3):
    # we do so by assuming the variance corresponds to an uniform distribution.
    # var(X) = (b - a)^2 / 12
    # mu(X) = (a+b) / 2
    # doing some algebra gives us
    # b = mu + \sqrt(3*var)
    # a = mu - \sqrt(3*var)
    sqrt_z_var = _np.sqrt(z*P_var)
    return {"p_up": P + sqrt_z_var, "p_low": P - sqrt_z_var}


def create_problem_list(options_object, problems_dict):
    # create a list of problems that indicate what problems are used for evaluation.
    # output should be a list of problems.
    # P is the transition kernel, R the reward kernel,
    # t_max how many steps are taken,
    # P_var the variance (uncertainty) of the true transition kernel.

    environment_format = problems_dict["format"]
    result = []

    if environment_format == "list":
        for problem in problems_dict["list"]:
            problem_type = problem["type"]
            problem_parameters = retrieve_from_dict(problem, "parameters", {})

            problem_to_add = {
                "problem_name": problem_type,
                "parameters": problem_parameters
            }

            if problem_type == "forest_default":
                P, R = mdptoolbox.example.forest()
                variance = retrieve_from_dict(problem_parameters, "variance", 0.05)
                P_var = _np.full(P.shape, variance)
                problem_to_add.update({"P": P, "R": R, "P_var": P_var, "t_max": options_object["t_max_def"]})

            elif problem_type == "forest":
                # number of states
                S = retrieve_from_dict(problem_parameters, "S", 10)
                # reward when 'wait' is performed in its oldest state
                r1 = retrieve_from_dict(problem_parameters, "r1", 40)
                # reward when 'cut' is performed in its oldest state
                r2 = retrieve_from_dict(problem_parameters, "r2", 1)
                # probability of fire
                p = retrieve_from_dict(problem_parameters, "p", 0.05)
                variance = retrieve_from_dict(problem_parameters, "variance", 0.05)

                P, R = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)
                # S x A -> A x S x S'
                R = _np.transpose(R)
                R = _np.repeat(R[:, :, _np.newaxis], R.shape[1], axis=2)

                P_var = _np.full(P.shape, variance)
                problem_to_add.update({"P": P, "R": R, "P_var": P_var, "t_max": options_object["t_max_def"]})

            elif problem_type == "random_highest_p":
                # create random mdp
                # adds uncertainty to most probable event for each state-action
                S = retrieve_from_dict(problem_parameters, "S", 10)
                A = retrieve_from_dict(problem_parameters, "A", 3)
                variance = retrieve_from_dict(problem_parameters, "variance", 0.05)
                P, R = mdptoolbox.example.rand(S, A, is_sparse=False)
                # most probable event per action, state
                P_argmax = _np.argmax(P, 2)
                P_var = _np.zeros(P.shape)

                for action_index in range(A):
                    for state_index in range(S):
                        P_var[action_index, state_index, P_argmax[action_index, state_index]] = variance

                # add to set
                problem_to_add.update({"P": P, "R": R, "P_var": P_var, "t_max": options_object["t_max_def"]})

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
    # I choose "as part of the mdp" for now
    discount_factor = retrieve_from_dict(mdp_hyperparameters, "discount_factor", 0.9)

    mdp_out = None
    # define mdp_out based on the type and any hyperparameters
    if mdp_type == "randomMdp":
        mdp_out = mdp_base.RandomMdp(P, R, None, None, None, None)
    elif mdp_type == "robustInterval":
        interval = compute_interval_by_variance(
            P, problem["P_var"], retrieve_from_dict(mdp_hyperparameters, "z", 3)
        )
        mdp_out = mdptoolbox.Robust.RobustIntervalModel(
            P, R, discount=discount_factor,
            p_lower=interval["p_low"], p_upper=interval["p_up"])
    elif mdp_type == "valueIteration":
        mdp_out = mdptoolbox.mdp.ValueIteration(P, R, discount=discount_factor)

    mdp_out.run()
    return mdp_out


def evaluate_mdp_results(result_mdp, options):
    # take the results + options object to define how we want to log

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


def air_conditioning_problem():
    # todo: implement this method
    # todo: (for later) create a continuous version instead of a fixed one,
    #  such that we can do simulation based upon calling functions
    # fixed time horizon
    # we can set it to 1 to simulate only computing the next action to take (right?)
    T = 1
    t = range(T)
    # fixed set of temperatures
    s = _np.arange(18, 23, 0.1)
    # indoor temperature
    x = _np.ones([1, T])
    # control input
    # a discrete version would be that we can either wait (0) or start air conditioning (1)
    # a continuous version would have a target temperature
    u = _np.array([0, 1])
    # disturbance, drawn from N(2, 0.2)
    w = _np.random.normal(2, 0.2, T)
    # our optimal temperature
    s_opt = 20.5
    s_more_opt = s >= s_opt

    r1 = _np.random.normal(2, 0.2, len(s))\
        - _np.multiply(_np.exp(s - s_opt), s_more_opt)\
        - _np.multiply(_np.exp(s_opt - s), ~s_more_opt)

    discount_factor = 0.95
    # reward of not switching AC on
    r_wait = 0.01

    A = len(u)
    S = len(s)

    # construct reward function
    R = _np.zeros([A, S, S, T])
    for i in range(A):
        for ii in range(S):
            reward_found = _np.cross(
                _np.power(discount_factor, t),
                r1[ii] + _np.multiply(_np.random.normal(2000, 200), _np.multiply(u[i], r_wait))
            )
            # repeat over all states to get S x T matrix
            rr = _np.repeat(reward_found[_np.newaxis, :], S, axis=0)
            R[i, ii, :, :] = rr

    # todo: define p
    P = _np.zeros([A, S, S, T])

    # x[t+1] = k * x[t] + (1 - k)*(Theta - eta * R * P * u[t]) + w[t]
    # r1(s) = w - exp(s - s_opt) * np.ones(1 where s >= s_opt, else 0)
    #         - exp(s_opt - s) * np.ones(1 where s <= s_opt, else 0)
    # r2(a) = - ca (models the cost of air conditioning)
    # r[t,s,a] = 0.95^t * (r1[s] + n(2000, 200) * r2(a)

    # from the paper:
    # The samples of the transition probability vectors are constructed by adding a normally distributed random
    # variable with a mean of 0.05 and a standard deviation of 0.01 to the largest element of each column of
    # the original transition probability matrix <- see random_robust_mdp for a similar implementation


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
                "discount_factor": 0.9,
                # define z, in the equation "mu +/- sqrt(z*var)" for defining p_low and p_up.
                # By default z=3 (corresponds to a uniform distribution)
                "sigma_interval_factor": 3
            }
        },
        # {
        #     "type": "randomMdp",
        #     "parameters": {}
        # },
        {
            "type": "valueIteration",
            "parameters": {}
        }
    ],
    number_of_runs=100,
    options={
        "t_max_def": 100,
        "save_figures": True,
        "logging_behavior": "default",
        # "log_filename": "last_session.log"
        # "figure_save_path": "../../figures"
        "plot_disabled": False,
    },
    problems_dict={
        "format": "list",
        # "list": ["forest_default", "forest_risky"],
        "list": [
            {
                "type": "random_highest_p",
                "parameters": {
                    "S": 10,
                    "A": 5,
                    "variance": 0.05
                }
            },
            {
                "type": "forest",
                "parameters": {}
            }
        ],
    }
)

