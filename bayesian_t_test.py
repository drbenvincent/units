import pymc3 as pm


def make_equal_variance_model(pdata, target_variable):
    '''Bayesian 2-group t-test, equal variance model.'''
    
    # decant data
    group = pdata['group']
    x = pdata[pdata.group==0][target_variable].values
    y = pdata[pdata.group==1][target_variable].values
    n1, n2 = sum(group==0), sum(group==1)
    n_groups = 2
    σ_low, σ_high = 0.0001, 10

    with pm.Model() as model:

        # priors
        group_means = pm.Normal('group_means', mu=[-2, -4], sd=5, shape=n_groups)
        group_std = pm.Uniform('group_std', lower=σ_low, upper=σ_high)

        # compute effect size

        effect_size = pm.Deterministic('effect size', (group_means[0] - group_means[1]) /
                                       np.sqrt(
                                           ((n1-1)*group_std**2 + (n2-1)*group_std**2)
                                           /(n1 + n2 - 2)
                                       ))
        # likelihood functions
        x = pm.Normal('x', mu=group_means[0], sd=group_std, observed=x)
        y = pm.Normal('y', mu=group_means[1], sd=group_std, observed=y)
    
    return model 


def make_unequal_variance_model(pdata, target_variable):
    '''Bayesian 2-group t-test, unequal variance model.'''
    
    # decant data
    group = pdata['group']
    x = pdata[pdata.group==0][target_variable].values
    y = pdata[pdata.group==1][target_variable].values
    n1, n2 = sum(group==0), sum(group==1)
    n_groups = 2
    σ_low, σ_high = 0.0001, 10

    with pm.Model() as model:

        # priors
        group_means = pm.Normal('group_means', mu=[-2, -4], sd=5, shape=n_groups)
        group_std = pm.Uniform('group_std', lower=σ_low, upper=σ_high, shape=n_groups)

        # compute effect size

        effect_size = pm.Deterministic('effect size', (group_means[0] - group_means[1]) / 
                                       np.sqrt(
                                           ((n1-1)*group_std[0]**2 + (n2-1)*group_std[1]**2)
                                           / (n1 + n2 - 2)
                                       ))
        # likelihood functions
        x = pm.Normal('x', mu=group_means[0], sd=group_std[0], observed=x)
        y = pm.Normal('y', mu=group_means[1], sd=group_std[1], observed=y)
    
    return model      