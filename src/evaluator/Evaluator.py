import json
import os
from datetime import datetime
import numpy as _np
from matplotlib import pyplot
import seaborn as sns
from enum import Enum


class Sampling(Enum):
    ALL = 0
    IN = 1
    OUT = 2


class EM(Enum):
    COMPUTED = 0
    SIMULATED = 1


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
        self.distances = {}
        self.filter_ratio = {}
        self.logList = {}
        self.figures = {}
        self.time = {}

    """
    Run the destructor
    closes the log file and pyplot
    """
    def __del__(self):
        self.file_to_write.close()
        pyplot.close()

    def log(self, problem, mdp_key, sampling, evaluationMethod, results, distances, filter_ratio=None):
        self.results[problem, mdp_key, sampling, evaluationMethod] = results
        self.distances[problem, mdp_key, sampling, evaluationMethod] = distances
        print("logging " + str(len(results)) + " results and " + str(len(distances)) + " distances")
        self.filter_ratio[problem, mdp_key, sampling, evaluationMethod] = filter_ratio

    def write_log(self, problem, mdp_key, mdp):
        to_write = {
            "problem": self.problems[problem].getName(),
            "mdp": mdp.getName(),
            "time": self.time[problem, mdp_key],
            "policy": str(mdp.policy)}

        for sampling in Sampling:
            for evaluationMethod in EM:
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

                # log time that it took
                self.time[problem_key, mdp_key] = mdp.time

                # see if we need to evaluate on all results
                if self.options.evaluate_all:
                    if self.options.do_computation:
                        self.log(problem_key, mdp_key, Sampling.ALL, EM.COMPUTED,
                                 ps.computeMDP(mdp), ps.distances)
                    if self.options.do_simulation:
                        self.log(problem_key, mdp_key, Sampling.ALL, EM.SIMULATED,
                                 ps.simulateMDP(mdp), ps.distances)

                # see if we need inner or outer samples
                if self.options.evaluate_inner or self.options.evaluate_outer:
                    self.inner_samples, self.outer_samples = ps.split(mdp)

                # see if we need to evaluate on inner results
                if self.options.evaluate_inner:
                    filter_ratio = len(self.inner_samples.samples) / len(ps.samples)
                    if self.options.do_computation:
                        self.log(problem_key, mdp_key, Sampling.IN, EM.COMPUTED,
                                 self.inner_samples.computeMDP(mdp), self.inner_samples.distances, filter_ratio)
                    if self.options.do_simulation:
                        self.log(problem_key, mdp_key, Sampling.IN, EM.SIMULATED,
                                 self.inner_samples.simulateMDP(mdp), self.inner_samples.distances, filter_ratio)

                # see if we need to evaluate on outer results
                if self.options.evaluate_outer:
                    filter_ratio = len(self.outer_samples.samples) / len(ps.samples)
                    if self.options.do_computation:
                        self.log(problem_key, mdp_key, Sampling.OUT, EM.COMPUTED,
                                 self.outer_samples.computeMDP(mdp), self.outer_samples.distances, filter_ratio)
                    if self.options.do_simulation:
                        self.log(problem_key, mdp_key, Sampling.OUT, EM.SIMULATED,
                                 self.outer_samples.simulateMDP(mdp), self.outer_samples.distances, filter_ratio)

                # write log
                self.write_log(problem_key, mdp_key, mdp)

            self.plot(problem_key)

    def plot(self, problem_key):
        for sampling in Sampling:
            for evaluationMethod in EM:
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

                results = {}
                for mdp_key, mdp_constructor in enumerate(self.mdpconstructors):
                    if (problem_key, mdp_key, sampling, evaluationMethod) in self.results:
                        name = mdp_constructor(self.problems[problem_key].transition_kernel,
                                               self.problems[problem_key].reward_matrix,
                                               self.problems[problem_key].discount_factor).getName()
                        # print("plotting " + name + " with " + str(sampling) + " " + str(evaluationMethod))
                        results[name] = self.results[problem_key, mdp_key, sampling, evaluationMethod]
                        if len(results[name]) > 0:
                            sns.distplot(results[name], hist=self.options.plot_hist, label=name)

                pyplot.legend()
                pyplot.savefig(self.log_dir + title + ".png", num=self.figures[problem_key, sampling, evaluationMethod],
                               dpi=150, format="png")
                pyplot.show()

                self.figures[problem_key, sampling, evaluationMethod, "scatter"] = pyplot.figure()
                title = self.problems[problem_key].getName() + "-" + str(sampling) + "-" + str(evaluationMethod)

                pyplot.title(title)
                pyplot.xlabel("Transition kernel distance")
                pyplot.ylabel("Value")


                results = {}
                distances = {}
                for mdp_key, mdp_constructor in enumerate(self.mdpconstructors):
                    if (problem_key, mdp_key, sampling, evaluationMethod) in self.results and \
                            (problem_key, mdp_key, sampling, evaluationMethod) in self.distances:
                        name = mdp_constructor(self.problems[problem_key].transition_kernel,
                                               self.problems[problem_key].reward_matrix,
                                               self.problems[problem_key].discount_factor).getName()
                        results[name] = self.results[problem_key, mdp_key, sampling, evaluationMethod]
                        distances[name] = self.distances[problem_key, mdp_key, sampling, evaluationMethod]
                        l = min(len(results[name]), len(distances[name]))
                        sns.scatterplot(x=distances[name][1:l], y=results[name][1:l], s=10, label=name)

                pyplot.legend()
                pyplot.savefig(self.log_dir + title + "scatter.png", num=self.figures[problem_key, sampling, evaluationMethod],
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
