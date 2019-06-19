import json
import os
from datetime import datetime
import numpy as _np
from matplotlib import pyplot
import seaborn as sns

class Evaluator(object):
    """
    create an evaluator, which can then be run.
    The arguments are dictionary objects.
    Opens a log file
    """
    def __init__(self, problems, mdpconstructors, options):
        # if there is a single problem make it a list
        if not isinstance(problems, list):
            self.problems = [problems]
        else:
            self.problems = problems

        # if there is a single MDPConstructor make it a list
        if not isinstance(mdpconstructors, list):
            self.mdpconstructors = [mdpconstructors]
        else:
            self.mdpconstructors = mdpconstructors

        self.options = options

        # find out where to log and make corosponding folders
        self.log_dir = "../../logs/" + datetime.now().strftime('%Y%m%d-%H%M%S') + "/"
        log_filename = self.log_dir + "results.log"
        os.makedirs(self.log_dir)

        # create log file
        self.file_to_write = open(log_filename, "w+")

        # we have no results so far
        self.results = None

    """
    Run the destructor
    closes the log file
    """
    def __del__(self):
        self.file_to_write.close()

    """
    Run the evaluator
    """
    def run(self):
        # a place to store the results
        results = {}

        # for all problems
        for problem_key, problem in enumerate(self.problems):
            # create a set with transition kernels similar to problem (as specified by options)
            ps = problem.getProblemSet(self.options)

            # for all mdp's
            for mdp_key, mdp_constructor in enumerate(self.mdpconstructors):

                # build mdp for problem
                mdp = mdp_constructor(problem.transition_kernel, problem.reward_matrix, problem.discount_factor)

                # output to the console
                print("Creating and evaluating " + str(mdp.getName()) + " for " + str(problem.getName()) + " problem")

                # run mdp
                mdp.run()

                # evaluate mdp on problem set
                results[problem_key, mdp_key], filter_ratio = ps.evaluate(mdp)
                self.log_results(problem_key, mdp_key, mdp.policy, results[problem_key, mdp_key], filter_ratio)

        self.plot_results(results)
        self.file_to_write.close()

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
        # legend = []

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



