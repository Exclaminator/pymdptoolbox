from enum import Enum
import json

class LoggingBehavior(Enum):
    DEFAULT = 0
    TABLE = 1


class Options(object):

    number_of_paths = 1
    t_max = 10
    number_of_sims = 10
    sample_var = 0.05
    sample_amount = 10000
    sample_uniform = True
    variance_scaling = False
    variance_lower = 0
    variance_upper = 255

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

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
