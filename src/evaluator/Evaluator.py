import json
import os
import random
from datetime import datetime
import numpy as _np
from matplotlib import pyplot
import seaborn as sns
from enum import Enum
from Options import LoggingBehavior
import pandas as pd


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

        # make sure no paths are skipped when using variance scaling.
        if self.options.variance_scaling:
            self.options.number_of_paths = self.options.sample_amount

        # find out where to log and make corosponding folders
        self.log_dir = "../../logs/" + datetime.now().strftime('%Y%m%d-%H%M%S') + "/"
        log_filename = self.log_dir + "results.log"
        os.makedirs(self.log_dir)

        # create log file
        self.file_to_write = open(log_filename, "w+")

        # log options
        # self.file_to_write.write(self.options.toJSON() + "\n")

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

        if self.options.logging_behavior == LoggingBehavior.TABLE:
            to_write_str = ""
            for sampling in Sampling:
                for evaluationMethod in EM:
                    if (problem, mdp_key, sampling, evaluationMethod) in self.results:
                        values = self.results[problem, mdp_key, sampling, evaluationMethod]
                        if len(values) == 0:
                            continue
                        name = mdp.getName() + "-" + str(sampling) + "-" + str(evaluationMethod)
                        average_value = _np.mean(values)
                        variance = _np.var(values)
                        lowest_value = _np.min(values)
                        delimiter = "\t"
                        row = name + delimiter \
                            + str(self.filter_ratio[problem, mdp_key, sampling, evaluationMethod]) + delimiter \
                            + str(lowest_value) + delimiter \
                            + str(average_value) + delimiter \
                            + str(variance) + delimiter \
                            + str(mdp.policy) + delimiter \
                            + str(self.time[problem, mdp_key]) + "\n"
                        to_write_str += row

        elif self.options.logging_behavior == LoggingBehavior.DEFAULT:
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
                        write_key = str(sampling) + "-" + str(evaluationMethod)
                        to_write[write_key] = {
                            "average_value": average_value,
                            "variance": variance,
                            "lowest_value": lowest_value,
                            "sample_size": len(values)
                        }
                        fr = self.filter_ratio[problem, mdp_key, sampling, evaluationMethod]
                        if fr is not None:
                            to_write[write_key]["filter_ratio"] = fr

            to_write_str = json.dumps(to_write, indent=4, separators=(',', ': '))
            to_write_str += "\n"
        else:
            err = "invalid logging behavior:"+str(self.options.logging_behavior)
            to_write_str = err

        self.file_to_write.write(to_write_str)


    """
    Run the evaluator
    """
    def run(self):
        # for all problems
        for problem_key, problem in enumerate(self.problems):
            # create a set with transition kernels similar to problem (as specified by options)
            if self.options.evaluate_all or self.options.evaluate_inner or self.options.evaluate_outer:
                ps = problem.getProblemSet(self.options)

            # for all mdp's
            for mdp_key, mdp_constructor in enumerate(self.mdpconstructors):
                try:
                    #needed for the split, but run is not needed
                    mdp = mdp_constructor(problem.transition_kernel, problem.reward_matrix, problem.discount_factor)
                    # output to the console what we are doing
                    print("Creating and evaluating " + str(mdp.getName()) + " for " + str(problem.getName()) + " problem")

                    # build mdp for problem
                    if not self.options.use_problem_set_for_policy:

                        # run mdp
                        mdp.run()
                        # log time that it took
                        self.time[problem_key, mdp_key] = mdp.time
                        print("Policy found in " + str(mdp.time) + " seconds")
                    else:
                        self.time[problem_key, mdp_key] = -1  # TODO: remove timing, or measure all generation times

                    total_sample_count = len(ps.samples)
                    number_of_paths = self.options.number_of_paths

                    # create inner and outer samples sets
                    if self.options.evaluate_inner or self.options.evaluate_outer:
                        self.inner_samples, self.outer_samples = ps.split(mdp)

                    # evaluate on all results
                    if self.options.evaluate_all:
                        ps_lim = ps.limit(number_of_paths)
                        if self.options.do_computation:
                            if not self.options.use_problem_set_for_policy:
                                result = ps_lim.computeMDP(mdp)
                            else:
                                result = ps_lim.computeMDPpolicyPerProblem(mdp_constructor, problem)
                            self.log(problem_key, mdp_key, Sampling.ALL, EM.COMPUTED,
                                     result, ps_lim.distances)
                        if self.options.do_simulation:
                            if not self.options.use_problem_set_for_policy:
                                result = ps_lim.simulateMDP(mdp)
                            else:
                                result = ps_lim.simulateMDP(mdp)
                            self.log(problem_key, mdp_key, Sampling.ALL, EM.SIMULATED,
                                     result, ps_lim.distances)

                    # evaluate on inner results
                    if self.options.evaluate_inner:
                        filter_ratio = len(self.inner_samples.samples) / total_sample_count
                        in_lim = self.inner_samples.limit(number_of_paths)
                        if self.options.do_computation:
                            if not self.options.use_problem_set_for_policy:
                                result = in_lim.computeMDP(mdp)
                            else:
                                result = in_lim.computeMDPpolicyPerProblem(mdp_constructor, problem)
                            self.log(problem_key, mdp_key, Sampling.IN, EM.COMPUTED,
                                     result, in_lim.distances, filter_ratio)
                        if self.options.do_simulation:
                            if not self.options.use_problem_set_for_policy:
                                result = in_lim.simulateMDP(mdp)
                            else:
                                result = in_lim.simulateMDPpolicyPerProblem(mdp_constructor, problem)
                            self.log(problem_key, mdp_key, Sampling.IN, EM.SIMULATED,
                                     result, in_lim.distances, filter_ratio)

                    # evaluate on outer results
                    if self.options.evaluate_outer:
                        filter_ratio = len(self.outer_samples.samples) / total_sample_count
                        out_lim = self.outer_samples.limit(number_of_paths)
                        if self.options.do_computation:
                            if not self.options.use_problem_set_for_policy:
                                result = out_lim.computeMDP(mdp)
                            else:
                                result = out_lim.computeMDPpolicyPerProblem(mdp_constructor, problem)
                            self.log(problem_key, mdp_key, Sampling.OUT, EM.COMPUTED,
                                     result, out_lim.distances, filter_ratio)
                        if self.options.do_simulation:
                            if not self.options.use_problem_set_for_policy:
                                result = out_lim.simulateMDP(mdp)
                            else:
                                result = out_lim.simulateMDPpolicyPerProblem(mdp_constructor, problem)
                            self.log(problem_key, mdp_key, Sampling.OUT, EM.SIMULATED,
                                     result, out_lim.distances, filter_ratio)

                    # write log
                    self.write_log(problem_key, mdp_key, mdp)
                except AssertionError as err:
                    print(err)
            self.plot(problem_key)
        self.scalability_analysis()

    def plot(self, problem_key):
        for sampling in Sampling:
            for evaluationMethod in EM:
                try:
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

                    for mdp_key, mdp_constructor in enumerate(self.mdpconstructors):
                        results = {}
                        color = {}
                        if (problem_key, mdp_key, sampling, evaluationMethod) in self.results:
                            name = mdp_constructor(self.problems[problem_key].transition_kernel,
                                                   self.problems[problem_key].reward_matrix,
                                                   self.problems[problem_key].discount_factor).getName()
                            # print("plotting " + name + " with " + str(sampling) + " " + str(evaluationMethod))
                            if len(self.results[problem_key, mdp_key, sampling, evaluationMethod]) > 0:
                                results[name] = self.results[problem_key, mdp_key, sampling, evaluationMethod]
                                color[name] = self.options.color[mdp_key]

                        min_length = _np.inf
                        for name in results:
                            min_length = min(min_length, len(results[name]))

                        holder = {}
                        for name in results:
                            holder[name] = random.sample(results[name], min_length)
                            sns.distplot(holder[name], color=color[name], hist=self.options.plot_hist, label=name)

                    pyplot.legend()
                    pyplot.savefig(self.log_dir + title + ".png", num=self.figures[problem_key, sampling, evaluationMethod],
                                   dpi=150, format="png")
                    pyplot.show()

                    # make scatterplot
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
                            leng = min(len(results[name]), len(distances[name]))
                            sns.scatterplot(x=distances[name][1:leng], y=results[name][1:leng],
                                            markers=self.options.marker[mdp_key], s=5, label=name)

                    pyplot.legend()
                    pyplot.savefig(self.log_dir + title + "_scatter.png", num=self.figures[problem_key, sampling,
                                                                                          evaluationMethod],
                                   dpi=150, format="png")
                    pyplot.show()

                    # make linear regression plot
                    self.figures[problem_key, sampling, evaluationMethod, "linregression"] = pyplot.figure()
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
                            leng = min(len(results[name]), len(distances[name]))
                            if leng >  0:
                                sns.regplot(x=distances[name][1:leng], y=results[name][1:leng], label=name,
                                            marker=self.options.marker[mdp_key])

                    pyplot.legend()
                    pyplot.savefig(self.log_dir + title + "_linear_regression.png",
                                   num=self.figures[problem_key, sampling, evaluationMethod], dpi=150, format="png")
                    pyplot.show()
                except ValueError as err:
                    print(err)

    def scalability_analysis(self):
        self.figures["scalability"] = pyplot.figure()
        fig, ax = pyplot.subplots()
        pyplot.title("Scalability Analysis")
        pyplot.xlabel("Instance size")
        pyplot.ylabel("Runtime in ms")

        runtimes = []
        sizes = []
        mdps = []
        for mdp_key, mdp_const in enumerate(self.mdpconstructors):
            p = self.problems[0]
            mdp = mdp_const(p.transition_kernel, p.reward_matrix, p.discount_factor)
            for problem_key, problem in enumerate(self.problems):
                if (problem_key, mdp_key) in self.time:
                    runtimes.append(self.time[problem_key, mdp_key])
                    sizes.append(_np.product(problem.transition_kernel.shape))
                    mdps.append(mdp.getName())
        df = pd.DataFrame(data={
            'runtimes': runtimes,
            'sizes': sizes,
            'mdp': mdps
        })
        self.file_to_write.write(df.to_csv())
        sns.pointplot(x='sizes', y='runtimes', hue="mdp", data=df)
        pyplot.yscale("log")
        pyplot.xscale("log")
        ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax.legend()
        pyplot.savefig(self.log_dir + "scalability_analysis.png",
                       num=self.figures["scalability"], dpi=150, format="png")
        pyplot.show()
