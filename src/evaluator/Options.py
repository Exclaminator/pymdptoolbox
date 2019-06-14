

class Options(object):

    # _default_dict = {
    #         NUMBER_OF_PATHS: 1,
    #         T_MAX: 10,
    #         NUMBER_OF_RUNS: 10,
    #         SAVE_FIGURES: True,
    #         LOGGING_BEHAVIOR: DEFAULT_KEY,
    #         NOISE_DIST: "gaussian",  # default: "gaussian" <- how the samples for ambiguity are drawn
    #         FIX_INTERVAL: True,  # perform checking such that p is always within the ambiguity set at simulation time.
    #         LOG_DIR: DEFAULT_KEY,
    #         FIGURE_PATH: DEFAULT_KEY,
    #         PLOT_DISABLED: False,
    #         DO_SIMULATION: False,
    #         DO_COMPUTATION: True,
    #         PLOT_HIST: False,
    #         EVALUATE_ALL: True,
    #         EVALUATE_INNER: True,
    #         EVALUATE_OUTER: True
    # }

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


# a = Options()
# a.is_default("number_of_paths")

# class Options(object):
#
#     # keys for the dict
#     number_of_paths = "number_of_paths"
#     t_max = "t_max_def"
#     number_of_runs = "number_of_runs"
#
#     save_figures = "save_figures"
#     logging_behavior = "logging_behavior"
#     fix_interval = "fix_interval"
#     log_dir = "log_dir"
#     figure_path = "figure_path"
#     plot_disabled = "plot_disabled"
#     do_simulation = "do_simulation"
#     do_computation = "do_computation"
#     noise_dist = "noise_dist"
#     plot_hist = "plot_hist"
#
#     # keys used for various objects
#     default_key = "default"
#     elements_key = "elements"
#     type_key = "type"
#     parameters_key = "parameters"
#
#     evaluate_all = "evaluate_all"
#     evaluate_inner = "evaluate_inner"
#     evaluate_outer = "evaluate_outer"
#
#
#
#     def __init__(self, options_dict):
#         # reads default, overwrites with options_dict
#         self.fields = {**Options._default_dict, **options_dict}
#
#     @staticmethod
#     def from_default():
#         return Options(Options._default_dict)
#
#     def is_default(self, field):
#         return self.get(field) == Options.DEFAULT_KEY
#
#     def get(self, field):
#         if field in self.fields:
#             return self.fields[field]
#         else:
#             return Options._default_dict[field]
