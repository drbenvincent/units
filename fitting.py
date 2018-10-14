from scipy.optimize import differential_evolution
import numpy as np


def MLE_procedure(func, bounds):
    """Our procedure for trying to ensure we find the globabl mimimum of the
    negative log likelihood.

    Note that `func` must be a function which takes parameters as the one and
    only argument. So if we are doing data fitting, which we are here, then we
    need to embedd the data into `func`

    example call:
        fit = MLE_procedure(func, data)
    """
    return differential_evolution(func, bounds)
