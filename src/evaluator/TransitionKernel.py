import abc
import numpy as _np
from scipy.stats import wasserstein_distance


def _normalizeP(P_in):
    Pout = _np.zeros(P_in.shape)
    for i in range(P_in.shape[0]):
        for ii in range(P_in.shape[1]):
            Pout[i, ii, :] = P_in[i, ii, :] / _np.sum(P_in[i, ii, :])

    return Pout


class TransitionKernel(object):
    """
    class for the transition kernel, often noted as P
    Used to define and simulate different types of ambiguity.
    ttk stands for the true transition kernel
    """
    def __init__(self, ttk):
        self.ttk = ttk

    @abc.abstractmethod
    def draw(self):
        """
        draws a transition kernel, according to its inner implementation.
        The transition kernel is a numpy array
        -------
        """
        return


class TransitionKernelInterval(TransitionKernel):

    """
    Use an upper and lower bound to define the ambiguity in P
    """
    def __init__(self, ttk, ttk_low, ttk_up):
        TransitionKernel.__init__(self, ttk)
        self.ttk_low = ttk_low
        self.ttk_up = ttk_up
        self.beta = self.compute_beta()

    def compute_beta(self):
        # add a beta, based on the maximum wasserstein distance possible wrt ttk_low and ttk_up
        shape = self.ttk.shape
        beta = _np.zeros(self.ttk.shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                beta[i, j] = _np.maximum(
                    wasserstein_distance(self.ttk[i, j], self.ttk_low[i, j]),
                    wasserstein_distance(self.ttk[i, j], self.ttk_up[i, j])
                )
        return _np.max(beta)
    """
    Use variance to construct P.
    We do so by assuming the variance corresponds to an uniform distribution.
    """
    @staticmethod
    def from_var(ttk, ttk_var):
        # var(X) = (b - a) ^ 2 / 12
        # mu(X) = (a + b) / 2
        # some algebra
        # b = mu + \sqrt(3 * var)
        # a = mu - \sqrt(3 * var)
        sqrt_z_var = _np.sqrt(3 * ttk_var)

        return TransitionKernelInterval(
            ttk,
            _np.maximum(ttk - sqrt_z_var, 0),
            _np.maximum(ttk + sqrt_z_var, 1)
        )

    """
    Draws from a uniform distribution
    """
    def draw(self):
        return _normalizeP(_np.random.uniform(self.ttk_low, self.ttk_up))


class TransitionKernelVar(TransitionKernel):

    def __init__(self, ttk, ttk_var):
        TransitionKernel.__init__(self, ttk)
        self.ttk_var = ttk_var

    """
    Transition kernel that uses a variance + distribution, to draw ambiguity from a gaussian
    """
    def draw(self):
        return _normalizeP(_np.random.normal(self.ttk, self.ttk_var))


class TransitionKernelBeta(TransitionKernel):

    """
    Uses beta to construct a transition matrix.
    Must be within wasserstein distance
    """
    def __init__(self, ttk, beta):
        TransitionKernel.__init__(self, ttk)
        self.beta = beta

    def draw(self):
        # todo: implement how to draw given some beta
        # todo: replace ttk by a drawn ttk
        return ttk
