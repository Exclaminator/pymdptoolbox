from Options import Options
import Problem
import os
import model_picker
from datetime import datetime
import numpy as _np
from matplotlib import pyplot
import seaborn as sns
import scipy as sp


class Evaluator(object):

    SIMULATED_KEY = "simulated_results"
    COMPUTED_KEY = "computed_results"

    """
    create an evaluator, which can then be run.
    The arguments are dictionary objects.
    """
    def __init__(self, problem_dict, options_dict):
        self.options = Options(options_dict)
        self.problem_list = self.parse_problem_dict(problem_dict)
        self.log_dir = None
        self.file_to_write = None

    """
    Run the evaluator
    """
    def run(self, mdp_set):

        if self.options.is_default(Options.LOG_DIR):
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        else:
            timestamp = self.options.get(Options.LOG_DIR)

        self.log_dir = "../../logs/" + timestamp + "/"
        log_filename = self.log_dir + "results.log"
        os.makedirs(self.log_dir)

        # create log file
        self.file_to_write = open(log_filename, "w+")

        for problem in self.problem_list:
            self.run_problem(problem, mdp_set[Options.ELEMENTS_KEY])

        self.file_to_write.close()

    @staticmethod
    def parse_problem_dict(problems_dict):
        result = []
        # the environment type can be used to define a different format than a list
        # at the moment we only support the "list" type
        for problem in problems_dict[Options.ELEMENTS_KEY]:
            result.append(Problem.create_problem_by_dict(problem))
        return result

    def run_problem(self, problem, mdp_list):
        # create P set
        p_set = []
        for i in range(self.options.get(Options.NUMBER_OF_PATHS)):
            p_set.append(problem.transition_kernel.draw())

        self.file_to_write.write(str(problem.KEY) + "\n")
        results_for_plotting = {}

        # compute mdp performance
        for mdp_dict in mdp_list:
            mdp = model_picker.create_mdp(mdp_dict, problem)
            mdp_key = Evaluator.create_mdp_key(mdp_dict)
            # do evaluation on results for this mdp and log it
            self.file_to_write.write(mdp_key+":\n")
            self.file_to_write.write("policy: "+str(mdp.policy)+"\n")
            # self.file_to_write.write(str(evaluate_mdp_results(results_mdp_dict))+"\n")
            # self.file_to_write.write("Value for original p: {} )\n".format(vp))

            # do simulation
            if self.options.get(Options.DO_SIMULATION):
                simulated_values = self.simulate_policy_on_problem(p_set, mdp.policy, problem)
                results_for_plotting[mdp_key, Evaluator.SIMULATED_KEY] = simulated_values
                self.result_summary_to_logfile(simulated_values)

            # do computation
            if self.options.get(Options.DO_COMPUTATION):
                computed_values = self.compute_policy_on_problem(p_set, mdp.policy, problem)
                results_for_plotting[mdp_key, Evaluator.COMPUTED_KEY] = computed_values
                self.result_summary_to_logfile(computed_values)

        # make plots
        if ~self.options.get(Options.PLOT_DISABLED):
            # retrieving the corresponding keys for the plots
            keys_tuples = list(results_for_plotting.keys())

            if self.options.get(Options.DO_SIMULATION):
                keys_simulated = list(filter(lambda x: x[1] == Evaluator.SIMULATED_KEY, keys_tuples))
                self.make_figure_plot(results_for_plotting, keys_simulated, problem.KEY + " simulated")

            if self.options.get(Options.DO_COMPUTATION):
                keys_computed = list(filter(lambda x: x[1] == Evaluator.COMPUTED_KEY, keys_tuples))
                self.make_figure_plot(results_for_plotting, keys_computed, problem.KEY + " computed")

    @staticmethod
    def create_mdp_key(mdp_dict):
        mdp_key = mdp_dict[Options.TYPE_KEY]

        if mdp_key == model_picker.ROBUST_MDP_KEY:
            mdp_key += "-"+mdp_dict[Options.PARAMETERS_KEY][model_picker.SIGMA_IDENTIFIER_KEY]

        return mdp_key

    def make_figure_plot(self, values, keys, title):
        legend = []
        results = [values[x] for x in keys]
        for i in range(len(results)):
            sns.distplot(results[i], hist=self.options.get(Options.PLOT_HIST), label=keys[i][0])
        #  legend.append(keys[i][0])

        pyplot.title(title)
        pyplot.xlabel("Value")
        pyplot.ylabel("Frequency")
        pyplot.legend()

        pyplot.savefig(self.log_dir + title + ".png", dpi=150, format="png")
        pyplot.show()
        pyplot.close()

    def result_summary_to_logfile(self, results):
        average_value = _np.mean(results)
        variance = _np.var(results)
        lowest_value = _np.min(results)

        # define logging behavior here
        # maybe add "verbose" or "minimal" etc.
        if self.options.is_default(Options.LOGGING_BEHAVIOR):
            to_write = {
                "average_value": average_value,
                "variance": variance,
                "lowest_value": lowest_value
            }
        else:
            to_write = {}
        self.file_to_write.write(str(to_write))

    @staticmethod
    def compute_policy_on_problem(p_set, policy, problem):
        values = []
        reward_matrix = problem.reward_matrix
        for P in p_set:

            # P and R are A x S x S' shaped
            S = len(policy)
            discount_factor = problem.discount_factor

            def computePPolicy(state):
                return P[policy[state], state, :]

            def computeRPolicy(state):
                return reward_matrix[policy[state], state, :]

            # hacky conversion using list (otherwise it will return non-numeric objects)
            P_arr = _np.array(list(map(computePPolicy, range(S))))
            R_arr = _np.array(list(map(computeRPolicy, range(S))))

            R_vector = _np.zeros(S)
            for i in range(S):
                for j in range(S):
                    R_vector[i] += R_arr[i][j] * P_arr[i][j]

            V = _np.linalg.solve(
                sp.eye(S) - discount_factor * P_arr,
                R_vector)

            values.append(V[0])
        return values

    @staticmethod
    def simulate_policy_on_problem(p_set, policy, problem, runs, t_max):
        values = []
        reward_matrix = problem.reward_matrix
        discount_factor = problem.discount_factor

        for P in p_set:
            values_for_P = []

            for i in range(runs):
                s = 0
                total_reward = 0

                for t in range(t_max):
                    action = policy[s]
                    P_a = P[action, s]
                    s_new = _np.random.choice(a=len(P_a), p=P_a)
                    # R is in format A x S x S'
                    RR = reward_matrix[:, s, s_new]
                    total_reward += RR[action] * _np.power(discount_factor, t)
                    s = s_new

                values_for_P.append(total_reward)
            values.append(_np.mean(values_for_P))

        return values


