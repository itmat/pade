from __future__ import print_function

import numpy as np
import statsmodels.api as sm
from collections import namedtuple
import time

def time_fn(f, *args, **kwargs):
    start = time.time()
    res = f(*args, **kwargs)
    end = time.time()
    return (end - start, res)

class GlmResults(object):
    def __init__(self, y, x, family, contrast, models, fitteds, fs):
        self.y = y
        self.x = x
        self.family = family
        self.contrast = contrast
        self.models = models
        self.fitteds = fitteds
        self.fs = fs

    @property
    def fittedvalues(self):
        return np.array([fitted.fittedvalues for fitted in self.fitteds])

    @property
    def params(self):
        return np.array([ fitted.params for fitted in self.fitteds])

    @property
    def weights(self):
        return np.array([ m.weights for m in self.models ])

    @property
    def f_values(self):
        return np.array([ f.fvalue[0,0] for f in self.fs ])
    
Results = namedtuple('Results', ['model', 'fitted', 'f_test'])

def old_glm(y, x, family, contrast):
    models = []
    fitteds = []
    fs = []
    for row in y:
        model = sm.GLM(row, x, family)
        fitted = model.fit()
        f = fitted.f_test(contrast)
        models.append(model)
        fitteds.append(fitted)
        fs.append(f)

    return GlmResults(y, x, family, contrast, models, fitteds, fs)

def new_glm(y, x, family, contrast):
    models = []
    fitteds = []
    fs = []

    model = VectorizedGLM(y.T, x, family)
    fitted = model.fit()
    f = fitted.f_test(contrast)

    return GlmResults(y, x, family, contrast, model, fitted, f)



def main():
    y = np.random.poisson(size=(100, 24))
    
    for i in range(30):
        y[i, :12] = y[i, :12] * np.random.poisson()

    for i in range(30, 60):
        y[i, 12:] = y[i, 12:] * np.random.poisson()

    x = np.zeros((24, 2), int)
    x[:, 0] = 1
    x[12:, 1] = 1

    contrast = [0, 1]

    (old_time, old_res) = time_fn(old_glm, y, x, sm.families.Poisson(), contrast)
    (new_time, new_res) = time_fn(new_glm, y, x, sm.families.Poisson(), contrast)

    np.testing.assert_almost_equal(old_res.params, new_res.params)
    np.testing.assert_almost_equal(old_res.fittedvalues, new_res.fittedvalues)
    np.testing.assert_almost_equal(old_res.weights, new_res.weights)
    np.testing.assert_almost_equal(old_res.f_values, new_res.f_values)

    print(old_time, new_time)



class VectorizedGLM(sm.GLM):    
    pass
    

if __name__ == '__main__':
    main()




