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
    sample_method = "uniform" #normal, uniform, monte carlo
    variance_scaling = False
    variance_lower = 0
    variance_upper = 255

    non_robust_actions = []

    color = [
        'tab:blue',
        'tab:orange',
        'tab:red',
#        'tab:purple',
#        'tab:brown',
#        'tab:pink',
        'tab:green',
        'tab:gray',
#        'tab:olive',
#         'tab:cyan',
        'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    marker = ['o', 'v', "1", "s", "p", "P", "*", "+", "x", "D", "X", "2", "3", "4"]

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

    use_problem_set_for_policy = False
    monte_carlo_sampling_init_count_value = 0
    monte_carlo_sampling_random_samples = 100

    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
