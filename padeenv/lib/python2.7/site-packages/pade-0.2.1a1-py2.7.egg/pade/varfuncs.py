"""
Variance functions for use with the link functions in statsmodels.family.links

This is based on statsmodels/genmod/families/family.py in the
statsmodels project: https://github.com/statsmodels/statsmodels. 

I have copied and modified the source from statsmodels in order to
optimize the algorithms for our purposes. PADE is interested in
fitting many models, one for each feature in the input (from tens of
thousands to millions). In a single job, all the models have the same
number of parameters, and that number is typically quite small, such
as 2. We also have a relatively small number of observations, since
each one represents a replicate of an experiment.

In order to fit these models quickly, we have changed many of the
statsmodels methods to add an additional dimension to parameters and
return values; this extra dimension representing the features. I
started to modify a forked version of the statsmodels project, with
the intention of contributing the changes back to statsmodels, but it
would have been impossible to make these changes in a way that did not
break the original statsmodels behavior. This is because many of the
functions already accept input of varying dimensionality, and behave
differently depending on how many dimensions their input arrays have.

The statsmodels license follows:

    Copyright (C) 2006, Jonathan E. Taylor
    All rights reserved.

    Copyright (c) 2006-2008 Scipy Developers.
    All rights reserved.

    Copyright (c) 2009-2012 Statsmodels Developers.
    All rights reserved.


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

      a. Redistributions of source code must retain the above copyright notice,
         this list of conditions and the following disclaimer.
      b. Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
      c. Neither the name of Statsmodels nor the names of its contributors
         may be used to endorse or promote products derived from this software
         without specific prior written permission.


    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.


"""

__docformat__ = 'restructuredtext'

import numpy as np

class VarianceFunction(object):
    """
    Relates the variance of a random variable to its mean. Defaults to 1.

    Methods
    -------
    call
        Returns an array of ones that is the same shape as `mu`

    Notes
    -----
    After a variance function is initialized, its call method can be used.

    Alias for VarianceFunction:
    constant = VarianceFunction()

    See also
    --------
    statsmodels.family.family
    """

    def __call__(self, mu):
        """
        Default variance function

        Parameters
        -----------
        mu : array-like
            mean parameters

        Returns
        -------
        v : array
            ones(mu.shape)
        """
        mu = np.asarray(mu)
        return np.ones(mu.shape, np.float64)

constant = VarianceFunction()
constant.__doc__ = """
The call method of constnat returns a constant variance, ie., a vector of ones.

constant is an alias of VarianceFunction()
"""

class Power(object):
    """
    Power variance function

    Parameters
    ----------
    power : float
        exponent used in power variance function

    Methods
    -------
    call
        Returns the power variance

    Formulas
    --------
    V(mu) = numpy.fabs(mu)**power

    Notes
    -----
    Aliases for Power:
    mu = Power()
    mu_squared = Power(power=2)
    mu_cubed = Power(power=3)
    """

    def __init__(self, power=1.):
        self.power = power

    def __call__(self, mu):
        """
        Power variance function

        Parameters
        ----------
        mu : array-like
            mean parameters

        Returns
        -------
        variance : array
            numpy.fabs(mu)**self.power
        """
        return np.power(np.fabs(mu), self.power)

mu = Power()
mu.__doc__ = """
Returns np.fabs(mu)

Notes
-----
This is an alias of Power()
"""
mu_squared = Power(power=2)
mu_squared.__doc__ = """
Returns np.fabs(mu)**2

Notes
-----
This is an alias of statsmodels.family.links.Power(power=2)
"""
mu_cubed = Power(power=3)
mu_cubed.__doc__ = """
Returns np.fabs(mu)**3

Notes
-----
This is an alias of statsmodels.family.links.Power(power=3)
"""

class Binomial(object):
    """
    Binomial variance function

    Parameters
    ----------
    n : int, optional
        The number of trials for a binomial variable.  The default is 1 for
        p in (0,1)

    Methods
    -------
    call
        Returns the binomial variance

    Formulas
    --------
    V(mu) = p * (1 - p) * n

    where p = mu / n

    Notes
    -----
    Alias for Binomial:
    binary = Binomial()

    A private method _clean trims the data by 1e-10 so that p is in (0,1)
    """

    tol = 1.0e-10

    def __init__(self, n=1):
        self.n = n

    def _clean(self, p):
        return np.clip(p, Binomial.tol, 1 - Binomial.tol)

    def __call__(self, mu):
        """
        Binomial variance function

        Parameters
        -----------
        mu : array-like
            mean parameters

        Returns
        -------
        variance : array
           variance = mu/n * (1 - mu/n) * self.n
        """
        p = self._clean(mu / self.n)
        return p * (1 - p) * self.n

binary = Binomial()
binary.__doc__ = """
The binomial variance function for n = 1

Notes
-----
This is an alias of Binomial(n=1)
"""

class NegativeBinomial(object):
    '''
    Negative binomial variance function

    Parameters
    ----------
    alpha : float
        The ancillary parameter for the negative binomial variance function.
        `alpha` is assumed to be nonstochastic.  The default is 1.

    Methods
    -------
    call
        Returns the negative binomial variance

    Formulas
    --------
    V(mu) = mu + alpha*mu**2

    Notes
    -----
    Alias for NegativeBinomial:
    nbinom = NegativeBinomial()

    A private method _clean trims the data by 1e-10 so that p is in (0,inf)
    '''

    tol = 1.0e-10

    def __init__(self, alpha=1.):
        self.alpha = alpha

    def _clean(self, p):
        return np.clip(p, NegativeBinomial.tol, np.inf)

    def __call__(self, mu):
        """
        Negative binomial variance function

        Parameters
        ----------
        mu : array-like
            mean parameters

        Returns
        -------
        variance : array
            variance = mu + alpha*mu**2
        """
        p = self._clean(mu)
        return mu + self.alpha*mu**2

nbinom = NegativeBinomial()
nbinom.__doc__ = """
Negative Binomial variance function.

Notes
-----
This is an alias of NegativeBinomial(alpha=1.)
"""
