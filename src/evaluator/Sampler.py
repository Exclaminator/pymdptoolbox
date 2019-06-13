import numpy as _np


def _normalize_tk(tk_in):
    tk_out = _np.zeros(tk_in.shape)
    for i in range(tk_in.shape[0]):
        for ii in range(tk_in.shape[1]):
            tk_out[i, ii, :] = tk_in[i, ii, :] / _np.sum(tk_in[i, ii, :])

    return tk_out


# mights skip the sampler due to static methods
class Sampler(object):

    def create_large_problem_set(self, problem_in, ambiquity_models, sample_amount):
        # use variance to create some "large" set of problems
        max_var = 0
        # can be done nicer with some kind of smart mapping probably instead
        for am in ambiquity_models:
            if am.var > max_var:
                max_var = am.var

        ttk = problem_in.ttk
        # draw from normal
        # index 1 and 2 are kernel indices, 3 is the sample index
        unnormalized_tks = _np.random.normal(_np.repeat(ttk[:, _np.newaxis], sample_amount), max_var)
        problems_out = []
        for i in range(sample_amount):
            tk = _normalize_tk(unnormalized_tks[:, :, i])
            new_problem = problem_in
            new_problem.ttk = tk
            problems_out.append(new_problem)

        return problems_out

