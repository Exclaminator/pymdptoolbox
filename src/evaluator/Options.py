

class Options(object):

    number_of_paths = 1
    t_max = 10
    number_of_runs = 10
    sample_var = 0.05
    sample_amount = 1000

    save_figures = True
    logging_behavior = None

    log_dir = None
    figure_path = None

    plot_disabled = False
    do_simulation = False
    do_computation = True
    plot_hist = False
    evaluate_all = True
    evaluate_inner = True
    evaluate_outer = False

    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)
