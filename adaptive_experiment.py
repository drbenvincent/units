import numpy as np
import pandas as pd
from model import get_response


def make_dataset(discount_func, true_params):
    """Set up options, create an experiment 'generator' object, then run
    trial. Returns a pandas dataframe of designs and responses. Each row is a
    trial."""
    delays = np.array([1, 2, 7, 14, 30, 30*3, 365, 365*5])
    trials_per_delay = 8
    max_trials = delays.size * trials_per_delay
    design_generator = frye_et_al_generator(DB_vec=delays, RB=100,
                                            trials_per_delay=trials_per_delay)

    # TODO: shouldn't have to give this max_trials)
    data = []
    chose_delayed = None
    for trial in range(max_trials):
        design = design_generator.send(chose_delayed)
        chose_delayed = get_response(design, discount_func, true_params)

        # merge design dictionary and response
        trial_data = design
        trial_data.update({"R": chose_delayed})

        data.append(trial_data)

    return pd.DataFrame(data)


def frye_et_al_generator(DB_vec=[7, 14, 30, 30*3, 30*6, 30*9, 365], RB=100.,
                         trials_per_delay=5):
    """A python generator which pumps out designs according to the method
    described by:
    Frye, C. C. J., Galizio, A., Friedel, J. E., DeHart, W. B., & Odum, A. L.
    (2016). Measuring Delay Discounting in Humans Using an Adjusting Amount
    Task. Journal of Visualized Experiments, (107), 1-8.
    http://doi.org/10.3791/53584
    """

    DA = 0
    post_choice_adjustment = 0.25
    chose_delayed_response = None

    for DB in DB_vec:
        # reset some variables at start of new delay
        RA = RB * 0.5
        post_choice_adjustment = 0.25

        for delay_count in range(trials_per_delay):
            if delay_count is 0:
                chose_delayed_response = yield {"DA": DA, "DB": DB, "RB": RB, "RA": RA}
            else:
                if chose_delayed_response:
                    RA = RA + (RB * post_choice_adjustment)
                else:
                    RA = RA - (RB * post_choice_adjustment)

                post_choice_adjustment /= 2
                chose_delayed_response = yield {"DA": DA, "DB": DB, "RB": RB, "RA": RA}


def plot_data(data, ax):
    D = data['R'] == 1
    I = data['R'] == 0
    ax.scatter(x=data['DB'][D], y=data['RA'][D]/data['RB'][D],
               c='k', edgecolors='k', label='chose delayed')
    ax.scatter(x=data['DB'][I], y=data['RA'][I]/data['RB'][I],
               c='w', edgecolors='k', label='chose immediate')
    ax.set_xlabel(r'$D(\rm{days})$')
    ax.set_ylabel(r'$f(D)$')
