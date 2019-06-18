import numpy as _np
from Problem import Problem


class ProblemSet(object):
    """
    representation of a collection of problems.
    """

    def __init__(self, all_samples, mdp_init):
        self.all_samples = all_samples
        self.mdp_init = mdp_init

    def filter(self, all_problems):
        # all_problems is a list instead of a ProblemSet object
        if hasattr(self.mdp_init, "innerfunction"):
            return [x for x in all_problems if self.mdp_init.innerfunction.inSample(x.transition_kernel)]
        else:
            # non robust model
            return all_problems


def normalize_tk(tk_in):
    tk_in = _np.abs(tk_in)
    tk_in = _np.minimum(tk_in, 1)
    tk_in = _np.maximum(tk_in, 0)

    tk_out = _np.zeros(tk_in.shape)
    for i in range(tk_in.shape[0]):
        for ii in range(tk_in.shape[1]):
            tk_out[i, ii, :] = tk_in[i, ii, :] / _np.sum(tk_in[i, ii, :])

    return tk_out


def create_large_problem_list(true_problem, variance, sample_amount):
    ttk = true_problem.transition_kernel
    # draw from normal
    # index 1 and 2 are kernel indices, 3 is the sample index
    non_normalized_tks = _np.random.normal(_np.repeat(ttk[:, :, :, _np.newaxis], sample_amount, axis=3), variance)

    problems_out = []
    for i in range(sample_amount):
        tk = normalize_tk(non_normalized_tks[:, :, :, i])
        new_problem = Problem(tk, true_problem.reward_matrix, true_problem.discount_factor)
        #new_problem.transition_kernel = tk
        problems_out.append(new_problem)

    return problems_out
