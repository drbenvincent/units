from scipy.stats import bernoulli, norm


def get_response(design, df, params):
    """Response determined by Bernoulli trials"""
    p = _prob_choose_delayed(design, df, params)
    return bernoulli.rvs(p)


def _prob_choose_delayed(design, df, params):
    decision_variable = _decision_variable(design, df, params)
    p_choose_delayed = _choice_function(decision_variable)
    return p_choose_delayed


def _decision_variable(design, df, params):
    VA = design['RA'] * df(design['DA'], params)
    VB = design['RB'] * df(design['DB'], params)
    decision_variable = VB - VA
    return decision_variable


def _choice_function(decision_variable, epsilon = 0.01, alpha = 4):    
    return epsilon + (1-2*epsilon) * norm.cdf(decision_variable/alpha)


def calc_log_likelihood(designs, df, params):
    p = _prob_choose_delayed(designs, df, params)
    log_likelihood = bernoulli.logpmf(designs.R, p)
    return sum(log_likelihood)