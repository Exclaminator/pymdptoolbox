import json
import os
from datetime import datetime
import numpy as _np
from matplotlib import pyplot
import seaborn as sns
from enum import Enum


class Sampling(Enum):
    ALL = "ALL"
    IN_SAMPLING = "IN_SAMPLING"
    OUT_SAMPLING = "OUT_SAMPLING"


class EvaluationMethod(Enum):
    COMPUTED = "COMPUTED"
    SIMULATED = "SIMULATED"


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
        self.inner_samples = None
        self.outer_samples = None
        self.results = {}
        self.filter_ratio = {}
        self.logList = {}
        self.figures = {}

    """
    Run the destructor
    closes the log file and pyplot
    """
    def __del__(self):
        self.file_to_write.close()
        pyplot.close()

    def log(self, problem, mdp, mdp_key, sampling, evaluationMethod, results, filter_ratio=None):
        self.results[problem, mdp_key, sampling, evaluationMethod] = results
        self.filter_ratio[problem, mdp_key, sampling, evaluationMethod] = filter_ratio

    def write_log(self, problem, mdp_key, mdp):
        to_write = {
            "problem": self.problems[problem].getName(),
            "mdp": mdp.getName(),
            "policy": str(mdp.policy)}

        for sampling in Sampling:
            for evaluationMethod in EvaluationMethod:
                if (problem, mdp_key, sampling, evaluationMethod) in self.results:
                    values = self.results[problem, mdp_key, sampling, evaluationMethod]
                    if len(values) == 0:
                        continue
                    average_value = _np.mean(values)
                    variance = _np.var(values)
                    lowest_value = _np.min(values)
                    to_write[str(sampling) + "-" + str(evaluationMethod)] = {
                        "average_value": average_value,
                        "variance": variance,
                        "lowest_value": lowest_value,
                        "sample_size": len(values),
                        "filter_ratio": self.filter_ratio[problem, mdp_key, sampling, evaluationMethod]
                    }

        to_write_str = json.dumps(to_write, indent=4, separators=(',', ': '))

        self.file_to_write.write(to_write_str + "\n")


    """
    Run the evaluator
    """
    def run(self):
        # for all problems
        for problem_key, problem in enumerate(self.problems):
            # create a set with transition kernels similar to problem (as specified by options)
            ps = problem.getProblemSet(self.options)

            # for all mdp's
            for mdp_key, mdp_constructor in enumerate(self.mdpconstructors):
                # build mdp for problem
                mdp = mdp_constructor(problem.transition_kernel, problem.reward_matrix, problem.discount_factor)

                # output to the console what we are doing
                print("Creating and evaluating " + str(mdp.getName()) + " for " + str(problem.getName()) + " problem")

                # run mdp
                mdp.run()

                # see if we need to evaluate on all results
                if self.options.evaluate_all:
                    if self.options.do_computation:
                        self.log(problem_key, mdp, mdp_key, Sampling.ALL, EvaluationMethod.COMPUTED, ps.computeMDP(mdp))
                    if self.options.do_simulation:
                        self.log(problem_key, mdp, mdp_key, Sampling.ALL, EvaluationMethod.SIMULATED, ps.simulateMDP(mdp))

                # see if we need inner or outer samples
                if self.options.evaluate_inner or self.options.evaluate_outer:
                    self.inner_samples, self.outer_samples = ps.split(mdp)

                # see if we need to evaluate on inner results
                if self.options.evaluate_inner:
                    filter_ratio = len(self.inner_samples.samples) / len(ps.samples)
                    if self.options.do_computation:
                        self.log(problem_key, mdp, mdp_key, Sampling.IN_SAMPLING, EvaluationMethod.COMPUTED,
                                 self.inner_samples.computeMDP(mdp), filter_ratio)
                    if self.options.do_simulation:
                        self.log(problem_key, mdp, mdp_key, Sampling.IN_SAMPLING, EvaluationMethod.SIMULATED,
                                 self.inner_samples.simulateMDP(mdp), filter_ratio)

                # see if we need to evaluate on outer results
                if self.options.evaluate_outer:
                    filter_ratio = len(self.outer_samples.samples) / len(ps.samples)
                    if self.options.do_computation:
                        self.log(problem_key, mdp, mdp_key, Sampling.OUT_SAMPLING, EvaluationMethod.COMPUTED,
                                 self.outer_samples.computeMDP(mdp), filter_ratio)
                    if self.options.do_simulation:
                        self.log(problem_key, mdp, mdp_key, Sampling.OUT_SAMPLING, EvaluationMethod.SIMULATED,
                                 self.outer_samples.simulateMDP(mdp), filter_ratio)

                # write log
                self.write_log(problem_key, mdp_key, mdp)

            self.plot(problem_key)

    def plot(self, problem_key):
        for sampling in Sampling:
            for evaluationMethod in EvaluationMethod:
                found = False
                for mdp_key, mdp_constructor in enumerate(self.mdpconstructors):
                    if (problem_key, mdp_key, sampling, evaluationMethod) in self.results:
                        found = True
                if not found: continue

                self.figures[problem_key, sampling, evaluationMethod] = pyplot.figure()
                title = self.problems[problem_key].getName() + "-" + str(sampling) + "-" + str(evaluationMethod)

                pyplot.title(title)
                pyplot.xlabel("Value")
                pyplot.ylabel("Frequency")
                pyplot.legend()

                results = {}
                for mdp_key, mdp_constructor in enumerate(self.mdpconstructors):
                    if (problem_key, mdp_key, sampling, evaluationMethod) in self.results:
                        results[str(mdp_constructor)] = self.results[problem_key, mdp_key, sampling, evaluationMethod]
                        sns.distplot(results[str(mdp_constructor)], hist=self.options.plot_hist, label=str(mdp_constructor))

                pyplot.legend()
                pyplot.savefig(self.log_dir + title + ".png", num=self.figures[problem_key, sampling, evaluationMethod],
                               dpi=150, format="png")
                pyplot.show()

    #
    #
    # def plot_results(self):
    #
    #     figures = {}
    #     # legend = []
    #
    #     # create all nesseceary plots
    #     for problem_key, problem in enumerate(self.problems):
    #         for sampling in Sampling:
    #             for evaluationMethod in EvaluationMethod:
    #                 found = False
    #                 for mdp_key, mdp_constructor in enumerate(self.mdpconstructors):
    #                     if (problem, mdp_key, sampling, evaluationMethod) in self.results:
    #                         found = True
    #                 if found:
    #                     figures[problem_key, sampling, evaluationMethod] = pyplot.figure()
    #
    #     for (problem_key, mdp_key), mp_result in results.items():
    #          for (set_key, evaluation_key), values in mp_result.items():
    #             if len(values) == 0:
    #                 continue
    #             # add figure to dict if not added
    #             if (problem_key, set_key, evaluation_key) not in figures.keys():
    #                 # initialize figure
    #                 figure = pyplot.figure()
    #                 figures[problem_key, set_key, evaluation_key] = figure
    #             else:
    #                 # set figure index
    #                 pyplot.figure(figures[problem_key, set_key, evaluation_key].number)
    #
    #             # plot to the figure which is initialized in the if statement above
    #             sns.distplot(values, hist=self.options.plot_hist, label=mdp_key)
    #
    #     for (problem_key, set_key, evaluation_key), figure in figures.items():
    #         # plot and show figure
    #         pyplot.figure(figure.number)
    #         title = problem_key + "-" + set_key + "-" + evaluation_key
    #         pyplot.title(title)
    #         pyplot.xlabel("Value")
    #         pyplot.ylabel("Frequency")
    #         pyplot.legend()
    #         pyplot.savefig(self.log_dir + title + ".png", num=figure, dpi=150, format="png")
    #
    #     pyplot.show()
