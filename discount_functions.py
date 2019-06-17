import numpy as np

# NOTE: for the sake of simplicity and clarity, we will be using the direct 
# parameterisation of k and kappa here. In other contexts it is advantageous to 
# fit log(k) or log(kappa) but we'll keep things simple here in order to not get confused.

def rachlin(delay, params):
    s, k = params
    assert s >= 0, 's is less than zero'
    return 1 / (1+k*delay**s)


def rachlin_kappa(delay, params):
    s, kappa = params
    assert s >= 0, 's is less than zero'
    assert kappa >= 0, 'kappa is less than zero'
    return 1 / (1+(kappa*delay)**s)


def hyperbolic_k(delay, params):
    k = params
    assert k >= 0, 'k is less than zero'
    return 1 / (1+k*delay)
