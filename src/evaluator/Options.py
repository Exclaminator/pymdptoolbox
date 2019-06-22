from enum import Enum


class LoggingBehavior(Enum):
    DEFAULT = 0
    TABLE = 1


class Options(object):

    number_of_paths = 1
    t_max = 10
    number_of_sims = 10
    sample_var = 0.5
    sample_amount = 1000
    sample_uniform = False
    non_robust_actions = []

    save_figures = True
    logging_behavior = LoggingBehavior.DEFAULT

    log_dir = None
    figure_path = None

    plot_disabled = False
    do_simulation = False
    do_computation = True
    plot_hist = False
    evaluate_all = True
    evaluate_inner = True
    evaluate_outer = True

    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)
