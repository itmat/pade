from __future__ import print_function

import unittest
import numpy as np
import pade.family as fam
import pade.glm as glm

class GlmTestCase(unittest.TestCase):

    def setUp(self):
        self.input = np.genfromtxt('pade/test/glm/in.txt')
        self.contrast = np.array([ [0, 1] ])
        self.exog = np.zeros((24, 2), int)
        self.exog[:, 0] = 1
        self.exog[12:, 1] = 1

        self.families = {
            'poisson'           : fam.Poisson(),
            'gaussian'          : fam.Gaussian(),
            'negative_binomial' : fam.NegativeBinomial(),
            'gamma'             : fam.Gamma()
            }

    def _run_family(self, family_name):

        family = self.families[family_name]
        (beta, mu, weights, cov_p, scale) = glm.fit_glm(self.input, self.exog, family)
        f = glm.f_test(beta, self.contrast, cov_p, scale)
        f = np.reshape(f, (len(f),))

        got = {
            'beta' : beta,
            'mu'   : mu,
            'weights' : weights,
            'f_values' : f }

        for (name, values) in got.items():
            print("Doing " + name)
            filename = 'pade/test/glm/{0}_{1}.txt'.format(family_name, name)
            expected = np.genfromtxt(filename)
            np.testing.assert_almost_equal(expected, values)


    def test_poisson(self):
        self._run_family('poisson')

    def test_gaussian(self):
        self._run_family('gaussian')

        
        
