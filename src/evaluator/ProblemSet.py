import numpy as _np
from Problem import Problem

class ProblemSet(object):
    """
    representation of a collection of problems.
    """

    def __init__(self, true_problem, mdp_init):
        self.true_problem = true_problem
        self.mdp_init = mdp_init
        if hasattr(mdp_init, "innerfunction"):
            self.innerfunction = mdp_init.innerfunction

    def filter(self, all_problems):
        # all_problems is a list instead of a ProblemSet object
        if hasattr(self.mdp_init, "innerfunction"):
            return [x for x in all_problems if self.mdp_init.innerfunction.inSample(x.transition_kernel)]
        else:
            # non robust model
            return all_problems
        # return all_problems.filter(self.mdp.innerfunction.inSample)


def _normalize_tk(tk_in):
    tk_out = _np.zeros(tk_in.shape)
    for i in range(tk_in.shape[0]):
        for ii in range(tk_in.shape[1]):
            tk_out[i, ii, :] = tk_in[i, ii, :] / _np.sum(tk_in[i, ii, :])

    return tk_out


# mights skip the sampler due to static methods

def create_large_problem_list(problem_set_in, variance, sample_amount):
    true_problem = problem_set_in.true_problem
    ttk = true_problem.transition_kernel
    # draw from normal
    # index 1 and 2 are kernel indices, 3 is the sample index
    non_normalized_tks = _np.random.normal(_np.repeat(ttk[:, :, :, _np.newaxis], sample_amount, axis=3), variance)
    non_normalized_tks = _np.minimum(non_normalized_tks, 1)
    non_normalized_tks = _np.maximum(non_normalized_tks, 0)

    problems_out = []
    for i in range(sample_amount):
        tk = _normalize_tk(non_normalized_tks[:, :, :, i])
        new_problem = Problem(tk, true_problem.reward_matrix, true_problem.discount_factor)
        #new_problem.transition_kernel = tk
        problems_out.append(new_problem)

    return problems_out
