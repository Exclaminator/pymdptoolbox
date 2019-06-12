

class Options(object):

    # keys for the dict
    NUMBER_OF_PATHS = "number_of_paths"
    T_MAX = "t_max_def"
    NUMBER_OF_RUNS = "number_of_runs"

    SAVE_FIGURES = "save_figures"
    LOGGING_BEHAVIOR = "logging_behavior"
    FIX_INTERVAL = "fix_interval"
    LOG_DIR = "log_dir"
    FIGURE_PATH = "figure_path"
    PLOT_DISABLED = "plot_disabled"
    DO_SIMULATION = "do_simulation"
    DO_COMPUTATION = "do_computation"
    NOISE_DIST = "noise_dist"
    PLOT_HIST = "plot_hist"
    # keys used for various objects
    DEFAULT_KEY = "default"
    ELEMENTS_KEY = "elements"
    TYPE_KEY = "type"
    PARAMETERS_KEY = "parameters"

    _default_dict = {
            NUMBER_OF_PATHS: 1,
            T_MAX: 10,
            NUMBER_OF_RUNS: 10,
            SAVE_FIGURES: True,
            LOGGING_BEHAVIOR: DEFAULT_KEY,
            NOISE_DIST: "gaussian",  # default: "gaussian" <- how the samples for ambiguity are drawn
            FIX_INTERVAL: True,  # perform checking such that p is always within the ambiguity set at simulation time.
            LOG_DIR: DEFAULT_KEY,
            FIGURE_PATH: DEFAULT_KEY,
            PLOT_DISABLED: False,
            DO_SIMULATION: False,
            DO_COMPUTATION: True,
            PLOT_HIST: False
    }

    def __init__(self, options_dict):
        # reads default, overwrites with options_dict
        self.fields = {**Options._default_dict, **options_dict}

    @staticmethod
    def from_default():
        return Options(Options._default_dict)

    def is_default(self, field):
        return self.get(field) == Options.DEFAULT_KEY

    def get(self, field):
        if field in self.fields:
            return self.fields[field]
        else:
            return Options._default_dict[field]
