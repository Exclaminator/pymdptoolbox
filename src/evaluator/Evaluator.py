import json
import ProblemSet
import os
from datetime import datetime
import numpy as _np
from matplotlib import pyplot
import seaborn as sns
import scipy as sp
import warnings

SIMULATED_KEY = "simulated"
COMPUTED_KEY = "computed"

ALL_KEY = "all"
INNER_KEY = "inner"
OUTER_KEY = "outer"
FILTER_RATIO_KEY = "filter_ratio"

def build_and_run(problem_dict, mdp_dict, options):
    Evaluator(options).run(problem_dict, mdp_dict)


class Evaluator(object):

    """
    create an evaluator, which can then be run.
    The arguments are dictionary objects.
    """
    def __init__(self, options):
        self.options = options

        self.file_to_write = None
        self.log_dir = None
        # we have no results so far
        self.results = None

    """
    Run the evaluator
    """
    def run(self, problem_dict, mdp_dict):

        if self.options.log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        else:
            timestamp = self.options.log_dir

        self.log_dir = "../../logs/" + timestamp + "/"
        log_filename = self.log_dir + "results.log"
        os.makedirs(self.log_dir)

        # create log file
        self.file_to_write = open(log_filename, "w+")

        results = {}

        for problem_key, problem in problem_dict.items():
            print(problem_key)
            all_samples = ProblemSet.create_large_problem_list(problem, self.options.sample_var,
                                                               self.options.sample_amount)
            for mdp_key, mdp in mdp_dict.items():
                print(mdp_key)
                mdp_init = mdp(problem.transition_kernel, problem.reward_matrix, problem.discount_factor)
                ps = ProblemSet.ProblemSet(all_samples, mdp_init)
                mdp_init.run()
                results[problem_key, mdp_key], filter_ratio = self.evaluate(ps, mdp_init.policy)
                self.log_results(problem_key, mdp_key, mdp_init.policy, results[problem_key, mdp_key], filter_ratio)

        self.plot_results(results)
        self.file_to_write.close()

    def evaluate(self, problem_set, policy):

        # Create large problem set
        all_samples = problem_set.all_samples

        # store results
        result = {}

        filter_ratio = 0

        if self.options.evaluate_all:
            result[ALL_KEY, COMPUTED_KEY], result[ALL_KEY, SIMULATED_KEY] = \
                self.evaluate_policy_on_problem_list(policy, all_samples)

        if self.options.evaluate_inner:
            inner_samples = problem_set.filter(all_samples)

            # store the ratio of filtered samples in results
            # result[INNER_KEY, FILTER_RATIO_KEY]\
            filter_ratio = len(inner_samples)/len(all_samples)

            result[INNER_KEY, COMPUTED_KEY], result[INNER_KEY, SIMULATED_KEY] = \
                self.evaluate_policy_on_problem_list(policy, inner_samples)

        # maybe only the outer samples are interesting.
        # If you think so, uncomment and implement something for _np.difference that works
        # if self.options.get(Options.EVALUATE_OUTER):
        #     outer_samples = _np.difference(all_samples, inner_samples)
        #     result[OUTER_KEY, COMPUTED_KEY], result[OUTER_KEY, SIMULATED_KEY] = \
        #         self.evaluate_policy_on_problem_list(policy, outer_samples)

        return result, filter_ratio

    def evaluate_policy_on_problem_list(self, policy, problem_list):
        # limit on the number of paths
        number_of_paths = self.options.number_of_paths
        if len(problem_list) > number_of_paths:
            problem_list = problem_list[0:number_of_paths]
        else:
            warnings.warn("number_of_paths ({}) is larger than the number of filtered policies ({})".format(number_of_paths, len(problem_list)))

        # use problem set to filter all problems
        results_computed = []
        results_simulated = []

        for problem in problem_list:
            # do this both for simulation and computation
            results_computed.append(self.compute_policy_on_problem(policy, problem))
            if self.options.do_simulation:
                results_simulated.append(self.simulate_policy_on_problem(policy, problem))

        return results_computed, results_simulated


    @staticmethod
    def compute_policy_on_problem(policy, problem):
        reward_matrix = problem.reward_matrix
        transition_kernel = problem.transition_kernel

        # P and R are A x S x S' shaped
        state_amount = len(policy)
        discount_factor = problem.discount_factor

        def compute_tk_policy(state):
            return transition_kernel[policy[state], state, :]

        def compute_rm_policy(state):
            return reward_matrix[policy[state], state, :]

        # hacky conversion using list (otherwise it will return non-numeric objects)
        tk_arr = _np.array(list(map(compute_tk_policy, range(state_amount))))
        rm_arr = _np.array(list(map(compute_rm_policy, range(state_amount))))

        rm_vector = _np.zeros(state_amount)
        for i in range(state_amount):
            for j in range(state_amount):
                rm_vector[i] += rm_arr[i][j] * tk_arr[i][j]

        V = _np.linalg.solve(
            sp.eye(state_amount) - discount_factor * tk_arr,
            rm_vector)

        return V[0]

    def simulate_policy_on_problem(self, policy, problem):
        reward_matrix = problem.reward_matrix
        discount_factor = problem.discount_factor
        tk = problem.transition_kernel

        results = []
        for i in range(self.options.number_of_sims):
            s_current = 0
            total_reward = 0

            for t in range(self.options.t_max):
                action = policy[s_current]
                tk_a = tk[action, s_current]
                s_new = _np.random.choice(a=len(tk_a), p=tk_a)
                # R is in format A x S x S'
                rm_3d = reward_matrix[:, s_current, s_new]
                total_reward += rm_3d[action] * _np.power(discount_factor, t)
                s_current = s_new

            results.append(total_reward)

        return _np.mean(results)

    def log_results(self, problem_key, mdp_key, policy, results, filter_ratio):
        # define logging behavior here
        # maybe add "verbose" or "minimal" etc.
        if self.options.logging_behavior is None:
            to_write = {
                "mdp": mdp_key,
                "problem": problem_key,
                "policy": str(policy),
                "filter_ratio": filter_ratio

            }
            for (set_key, eval_key), values in results.items():
                if len(values) == 0:
                    continue
                average_value = _np.mean(values)
                variance = _np.var(values)
                lowest_value = _np.min(values)
                to_write[set_key + "-" + eval_key] = {
                    "average_value": average_value,
                    "variance": variance,
                    "lowest_value": lowest_value,
                    "sample_size": len(values)
                }
            to_write_str = json.dumps(to_write, indent=4, separators=(',', ': '))
        else:
            to_write_str = "no logging due unsupported logging format: " + self.options.logging_behavior

        self.file_to_write.write(to_write_str + "\n")

    def plot_results(self, results):

        figures = {}
        legend = []

        for (problem_key, mdp_key), mp_result in results.items():
             for (set_key, evaluation_key), values in mp_result.items():
                if len(values) == 0:
                    continue
                # add figure to dict if not added
                if (problem_key, set_key, evaluation_key) not in figures.keys():
                    # initialize figure
                    figure = pyplot.figure()
                    figures[problem_key, set_key, evaluation_key] = figure
                else:
                    # set figure index
                    pyplot.figure(figures[problem_key, set_key, evaluation_key].number)

                # plot to the figure which is initialized in the if statement above
                sns.distplot(values, hist=self.options.plot_hist, label=mdp_key)

        for (problem_key, set_key, evaluation_key), figure in figures.items():
            # plot and show figure
            pyplot.figure(figure.number)
            title = problem_key + "-" + set_key + "-" + evaluation_key
            pyplot.title(title)
            pyplot.xlabel("Value")
            pyplot.ylabel("Frequency")
            pyplot.legend()
            pyplot.savefig(self.log_dir + title + ".png", num=figure, dpi=150, format="png")

        pyplot.show()
        pyplot.close()



