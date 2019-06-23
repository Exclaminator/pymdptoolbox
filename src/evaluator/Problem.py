import random

import numpy as _np
from numpy.core.defchararray import lower
from scipy.stats import wasserstein_distance

import mdptoolbox.example
import scipy as sp
from evaluator.Evaluator import Sampling
from mdptoolbox.Robust import Interval
import warnings

"""
creates a problem from a dict.
Selects the corresponding type
"""


def monte_carlo_sampling(transition_kernel, nr_kernels, init_count_value,
                         nr_samples_per_kernel):
    S = transition_kernel.shape[1]
    A = transition_kernel.shape[0]
    result = _np.zeros((nr_kernels, A, S, S))


    init_counts = []

    if init_count_value > 0:
        init_counts = _np.repeat(_np.array(range(S)), init_count_value)
    for i in range(nr_kernels):
        for a in range(A):
            for s in range(S):
                r = _np.random.choice(S, nr_samples_per_kernel, p = transition_kernel[a,s])
                ri = _np.append(r, init_counts)
                bins = range(0, S + 1)
                res, bins = _np.histogram(ri, bins= bins, density=True)
                result[i,a,s,:]= res

    res = _np.moveaxis(result, 0,3)
    return res


class Problem(object):
    """
    Creates a problem object. Uses the example module.
    We can initialize it based on some parameters.
    Next, we can extract P, R and the discount factor from it, which can be put into a MDP object.
    """

    def __init__(self, transition_kernel, reward_matrix, discount_factor=0.9, name="undefined", distance=0):
        self.transition_kernel = transition_kernel
        self.reward_matrix = reward_matrix
        self.discount_factor = discount_factor
        self.name = name
        self.distance = distance

    def getName(self):
        return self.name

    def getProblemSet(self, options):
        variance = options.sample_var
        sample_amount = options.sample_amount

        ttk = self.transition_kernel

        # index 1 and 2 are kernel indices, 3 is the sample index
        mu = _np.repeat(ttk[:, :, :, _np.newaxis], sample_amount, axis=3)

        # if we use variance scaling.
        # make a kernel for each variance between upper and lower limit
        if options.variance_scaling:
            variance = _np.divide(
                range(options.variance_lower, options.sample_amount),
                options.sample_amount / options.variance_upper)

        if lower(options.sample_method) == "uniform":
            # sample from uniform
            tk_low, tk_up = Interval.compute_interval(mu, variance)
            non_normalized_tks = _np.random.uniform(tk_low, tk_up)
        elif lower(options.sample_method) == "monte carlo":
            non_normalized_tks = monte_carlo_sampling(self.transition_kernel, sample_amount, options.monte_carlo_sampling_init_count_value, options.monte_carlo_sampling_random_samples)
        else:
            # sample from normal
            non_normalized_tks = _np.random.normal(mu, variance)


        problems_out = []
        for i in range(sample_amount):
            tk = self.normalize_tk(non_normalized_tks[:, :, :, i])
            for a in options.non_robust_actions:
                tk[a] = self.transition_kernel[a]

            distance = 0
            for a in range(self.transition_kernel.shape[0]):
                for s in range(self.transition_kernel.shape[1]):
                    distance += wasserstein_distance(tk[a][s], self.transition_kernel[a][s])
            new_problem = Problem(tk, self.reward_matrix, self.discount_factor, self.name, distance)
            # new_problem.transition_kernel = tk
            problems_out.append(new_problem)

        return ProblemSet(problems_out, self, options, Sampling.ALL)

    def computeMDP(self, mdp):
        policy = mdp.policy
        # P and R are A x S x S' shaped
        state_amount = len(policy)

        def compute_tk_policy(state):
            return self.transition_kernel[policy[state], state, :]

        def compute_rm_policy(state):
            return self.reward_matrix[policy[state], state, :]

        # hacky conversion using list (otherwise it will return non-numeric objects)
        tk_arr = _np.array(list(map(compute_tk_policy, range(state_amount))))
        rm_arr = _np.array(list(map(compute_rm_policy, range(state_amount))))

        rm_vector = _np.zeros(state_amount)
        for i in range(state_amount):
            for j in range(state_amount):
                rm_vector[i] += rm_arr[i][j] * tk_arr[i][j]

        V = _np.linalg.solve(
            sp.eye(state_amount) - self.discount_factor * tk_arr,
            rm_vector)

        return V[0]

    def simulateMDP(self, mdp, options):
        policy = mdp.policy

        results = []
        for i in range(options.number_of_sims):
            s_current = 0
            total_reward = 0

            for t in range(options.t_max):
                action = policy[s_current]
                tk_a = self.transition_kernel[action, s_current]
                s_new = _np.random.choice(a=len(tk_a), p=tk_a)
                # R is in format A x S x S'
                rm_3d = self.reward_matrix[:, s_current, s_new]
                total_reward += rm_3d[action] * _np.power(self.discount_factor, t)
                s_current = s_new

            results.append(total_reward)

        return _np.mean(results)

    @staticmethod
    def normalize_tk(tk_in):
        tk_in = _np.abs(tk_in)
        # todo: check if we want to do this, 1 is ensured by normaliztion,
        #  clipping it here will reduce its value in normaliztion
        # tk_in = _np.minimum(tk_in, 1)
        tk_in = _np.maximum(tk_in, 0)

        tk_out = _np.zeros(tk_in.shape)
        for i in range(tk_in.shape[0]):
            for ii in range(tk_in.shape[1]):
                tk_out[i, ii, :] = tk_in[i, ii, :] / _np.sum(tk_in[i, ii, :])
        return tk_out

    @staticmethod
    def get_forest_problem(S=10, discount_factor=0.9, r1=40, r2=20, p=0.05):
        tk, reward_matrix = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)

        # S x A -> A x S x S'
        reward_matrix = _np.transpose(reward_matrix)
        reward_matrix = _np.repeat(reward_matrix[:, :, _np.newaxis], reward_matrix.shape[1], axis=2)

        # create problem
        return Problem(tk, reward_matrix, discount_factor, "forest("+str(S)+")")

    @staticmethod
    def get_random_problem(S=10, A=5, discount_factor=0.9):
        tk, reward_matrix = mdptoolbox.example.rand(S, A, is_sparse=False)
        reward_matrix = _np.maximum(reward_matrix, 0)

        return Problem(tk, reward_matrix, discount_factor, "random("+str(S)+")")


class ProblemSet(object):
    """
    representation of a collection of problems.
    """
    def __init__(self, samples, problem, options, sampling=Sampling.ALL):
        self.samples = samples
        self.problem = problem
        self.options = options
        self.sampling = sampling
        self.resultsComputed = []
        self.resultsSimulated = []
        self.distances = []

        # for problem in self.samples[1:min(len(self.samples), self.options.number_of_paths)]:
        for problem in self.samples:
            self.distances.append(problem.distance)

    def split(self, mdp):
        # all_problems is a list instead of a ProblemSet object
        if hasattr(mdp, "innerfunction"):
            in_samples = ProblemSet(
                [x for x in self.samples if mdp.innerfunction.inSample(x.transition_kernel)],
                self.problem,
                self.options,
                Sampling.IN)
            out_samples = ProblemSet(
                [x for x in self.samples if not mdp.innerfunction.inSample(x.transition_kernel)],
                self.problem,
                self.options,
                Sampling.OUT)
            return in_samples, out_samples

        else:
            # non robust model
            return self, self

    def computeMDP(self, mdp):
        # for problem in self.samples[1:min(len(self.samples), self.options.number_of_paths)]:
        self.resultsComputed = []
        for problem in self.samples:
            self.resultsComputed.append(problem.computeMDP(mdp))
        return self.resultsComputed

    def computeMDPpolicyPerProblem(self, mdp_constructor, test_problem):
        # for problem in self.samples[1:min(len(self.samples), self.options.number_of_paths)]:
        self.resultsComputed = []
        for i, problem in enumerate(self.samples):
            print("Compute {}/{}".format(i, len(self.samples)))
            mdp = mdp_constructor(problem.transition_kernel, problem.reward_matrix, problem.discount_factor)
            mdp.run()
            self.resultsComputed.append(test_problem.computeMDP(mdp))
        return self.resultsComputed

    #  if self.options.do_simulation:
    def simulateMDP(self, mdp):
        # for problem in self.samples[1:min(len(self.samples), self.options.number_of_paths)]:
        self.resultsSimulated = []
        for problem in self.samples:
            self.resultsSimulated.append(problem.simulateMDP(mdp, self.options))
        return self.resultsSimulated

    def simulateMDPpolicyPerProblem(self, mdp_constructor, test_problem):
        # for problem in self.samples[1:min(len(self.samples), self.options.number_of_paths)]:
        self.resultsSimulated = []
        for i, problem in enumerate(self.samples):
            print("Simulate {}/{}".format(i, len(self.samples)))
            mdp = mdp_constructor(problem.transition_kernel, problem.reward_matrix, problem.discount_factor)
            mdp.run()
            self.resultsSimulated.append(test_problem.simulateMDP(mdp, self.options))
        return self.resultsSimulated

    def limit(self, number_of_paths):
        # returns a new problem set, with a limited set of samples
        # limit the problem set to a fixed amount of problems
        if len(self.samples) < number_of_paths:
            warnings.warn("number_of_paths ({}) is larger than the number of filtered policies ({})"
                          .format(number_of_paths, len(self.samples)))
            samples = self.samples
        else:
            samples = self.samples[random.sample(self.samples, number_of_paths)]

        return ProblemSet(samples, self.problem, self.options, self.sampling)
