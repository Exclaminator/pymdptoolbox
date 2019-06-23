from Evaluator import Evaluator
from Options import Options, LoggingBehavior
from mdptoolbox.Robust import *
from mdptoolbox.mdp import ValueIteration

from Problem import Problem


"""
default configuration, runs the forest problem on some models
"""


def run_default():
    # get tk_low and tk_up for the interval model

   # forest = Problem.get_forest_problem(S=10, discount_factor=0.9, r1=10, r2=2, p=0.05)
    forest = Problem.get_random_problem()
    tk_low1, tk_up1 = Interval.compute_interval(forest.transition_kernel, 0.08)
    tk_low2, tk_up2 = Interval.compute_interval(forest.transition_kernel, 0.04)
    tk_low3, tk_up3 = Interval.compute_interval(forest.transition_kernel, 0.2)
    tk_low4, tk_up4 = Interval.compute_interval(forest.transition_kernel, 0.3)
    tk_low5, tk_up5 = Interval.compute_interval(forest.transition_kernel, 0.4)
    tk_low6, tk_up6 = Interval.compute_interval(forest.transition_kernel, 0.5)
    tk_low7, tk_up7 = Interval.compute_interval(forest.transition_kernel, 0.01)
    tk_low8, tk_up8 = Interval.compute_interval(forest.transition_kernel, 1.0)
    tk_low9, tk_up9 = Interval.compute_interval(forest.transition_kernel, 3.0)

    # tk_low = (tk-0.5).clip(min=0)
    # tk_up = (tk+0.5).clip(max=1)

    was = [Robust(Wasserstein(0.1 * x)) for x in range(0, 25)]
    was.append(Robust(Wasserstein(20)))
    el = [Robust(Ellipsoid(0.05 * x)) for x in range(0, 25)]
    el.append(Robust(Ellipsoid(20)))
    like = [Robust(Likelihood(40 - 1.0 * x, 0.001)) for x in range(0, 100)]
    like.append(Robust(Likelihood(-100, 0.001)))
    like.append(Robust(Likelihood(100, 0.001)) )
    pb = [Interval.compute_interval(forest.transition_kernel, x * 0.002) for x in range(0,25)]

    inter = [Robust(Interval(pb[x][0], pb[x][1])) for x in range(0,25)]
    value = [ValueIteration]

    mdps = concatenate((was, el, inter, value, like))
    test = list(mdps)
    # problems can also be supplied as a list
    evaluator = Evaluator(
        [forest],
        test,
        Options(
            number_of_paths=1000, # set size of in and out set
            number_of_sims=1000,
            plot_hist=True,
            do_simulation=False,
            evaluate_all=True,
            evaluate_inner=False,
            evaluate_outer=False,
            sample_var=0.5,
            sample_amount=100,
            sample_method = "normal", #normal, uniform, monte carlo
            monte_carlo_sampling_init_count_value = 1,
            monte_carlo_sampling_random_samples = 20,
            use_problem_set_for_policy=False,
            non_robust_actions=[],  # replace with [1] if for action there should be no robustness

            variance_scaling=True,
            variance_lower=0,
            variance_upper=1,
            logging_behavior=LoggingBehavior.TABLE,

        ))
    evaluator.run()


run_default()
