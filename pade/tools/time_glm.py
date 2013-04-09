from __future__ import print_function

import pade.family as fam
from statsmodels.tools.tools import rank

import numpy as np

from collections import namedtuple
import pade.glm as glm
import time

def time_fn(f, *args, **kwargs):
    start = time.time()
    res = f(*args, **kwargs)
    end = time.time()
    return (end - start, res)

GlmResults = namedtuple(
    'GlmResults',
    ['y', 'x', 'family', 'contrast', 
     'params', 'fittedvalues', 'weights', 'f_values'])

def old_glm(y, x, family, contrast):

    import statsmodels.api as sm


    models = []
    fitteds = []
    fs = []
    for row in y:
        model = sm.GLM(row, x, family)
        fitted = model.fit()
        f = fitted.f_test(contrast)
        fs.append(f)
        models.append(model)
        fitteds.append(fitted)

    return GlmResults(y, x, family, contrast, 
                      [x.params for x in fitteds],
                      [x.fittedvalues for x in fitteds],
                      [x.weights for x in models],
                      [x.fvalue for x in fs])

def new_glm(y, x, family, contrast):
    models = []
    fitteds = []
    fs = []

    f = None

    (params, mu, weights, cov_p, scale) = glm.fit_glm(y, x, family)
    f = glm.f_test(params, contrast, cov_p, scale)

    return GlmResults(y, x, family, contrast, 
                      params, mu, weights, f)

def time_glm(y, x, old_family, new_family, contrast):
    (new_time, new_res) = time_fn(new_glm, y, x, new_family, contrast)
    (old_time, old_res) = time_fn(old_glm, y, x, old_family, contrast)    

    for i in range(len(old_res.params)):
        if sum(np.abs(old_res.params[i] - new_res.params[i])) > 0.001:
            print(i, old_res.params[i], new_res.params[i])

    np.testing.assert_almost_equal(old_res.params, new_res.params)
    np.testing.assert_almost_equal(old_res.fittedvalues, new_res.fittedvalues)
    np.testing.assert_almost_equal(old_res.weights, new_res.weights)
    np.testing.assert_almost_equal(old_res.f_values, new_res.f_values)

    print(old_family.__class__, new_family.__class__, old_time, new_time)

def main():

    y = np.genfromtxt('pade/test/glm_in.txt')

    x = np.zeros((24, 2), int)
    x[:, 0] = 1
    x[12:, 1] = 1

    contrast = np.array([ [0, 1] ])

    import statsmodels.api as sm


#    time_glm(y, x, sm.families.Gamma(), fam.Gamma(), contrast)

    time_glm(y, x, sm.families.Poisson(), fam.Poisson(), contrast)
    time_glm(y, x, sm.families.Gaussian(), fam.Gaussian(), contrast)
    time_glm(y, x, sm.families.NegativeBinomial(), fam.NegativeBinomial(), contrast)


if __name__ == '__main__':
    main()




