import abc
import numpy as _np


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
        return _np.random.uniform(self.ttk_low, self.ttk_up)


class TransitionKernelVar(TransitionKernel):

    def __init__(self, ttk, ttk_var):
        TransitionKernel.__init__(self, ttk)
        self.ttk_var = ttk_var

    """
    Transition kernel that uses a variance + distribution, to draw ambiguity from a gaussian
    """
    def draw(self):
        return _np.random.normal(self.ttk, self.ttk_var)


class TransitionKernelBeta(TransitionKernel):

    """
    Uses beta to construct a transition matrix.
    """
    def __init__(self, ttk, beta):
        TransitionKernel.__init__(self, ttk)
        self.beta = beta

    def draw(self):
        # todo: implement how to draw given some beta
        return
