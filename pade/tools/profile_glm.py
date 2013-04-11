from __future__ import print_function

import pade.family as fam
from statsmodels.tools.tools import rank

import numpy as np

from collections import namedtuple
import pade.glm as glm
import time
import cProfile
import pstats

def new_glm(y, x, family, contrast):
    models = []
    fitteds = []
    fs = []

    f = None

    (params, mu, weights, cov_p, scale) = glm.fit_glm(y, x, family)
    f = glm.f_test(params, contrast, cov_p, scale, smoothing=np.arange(10))


y = np.genfromtxt('pade/test/glm/in.txt')

x = np.zeros((24, 2), int)
x[:, 0] = 1
x[12:, 1] = 1

contrast = np.array([ [0, 1] ])

family = fam.NegativeBinomial()

if __name__ == '__main__':
    cProfile.run('new_glm(y, x, family, contrast)', sort='time', filename='prof')
    p = pstats.Stats('prof')
    p.strip_dirs().sort_stats('time').print_stats(100)



