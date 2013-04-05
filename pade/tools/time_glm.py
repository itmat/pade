import numpy as np
import statsmodels.api as sm
from collections import namedtuple


class OldResults(object):
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

    return OldResults(y, x, family, contrast, models, fitteds, fs)

def main():
    y = np.random.poisson(size=(10, 24))
    
    for i in range(3):
        y[i, :12] = y[i, :12] * np.random.poisson()

    for i in range(3, 6):
        y[i, 12:] = y[i, 12:] * np.random.poisson()

    x = np.zeros((24, 2), int)
    x[:, 0] = 1
    x[12:, 1] = 1

    contrast = [0, 1]

    old_res = old_glm(y, x, sm.families.Poisson(), contrast)

    print old_res.params
    print old_res.fittedvalues
    print old_res.weights
    print old_res.f_values

if __name__ == '__main__':
    main()




