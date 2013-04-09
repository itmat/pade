from __future__ import print_function

import pade.family as fam
from statsmodels.tools.tools import rank

import numpy as np

from collections import namedtuple
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

    model = VectorizedGLM(y, x, family)
    (params, mu, weights, cov_p) = model.fit()
    f = f_test(params, contrast, cov_p, model.scale)

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

    y = np.genfromtxt('data.txt')[:10]

    x = np.zeros((24, 2), int)
    x[:, 0] = 1
    x[12:, 1] = 1

    contrast = np.array([ [0, 1] ])

    import statsmodels.api as sm


    time_glm(y, x, sm.families.Gamma(), fam.Gamma(), contrast)

    time_glm(y, x, sm.families.Poisson(), fam.Poisson(), contrast)
    time_glm(y, x, sm.families.Gaussian(), fam.Gaussian(), contrast)
    time_glm(y, x, sm.families.NegativeBinomial(), fam.NegativeBinomial(), contrast)



#TODO: untested for GLMs?
def f_test(params, r_matrix, cov_p, scale):

    if (cov_p is None):
        raise ValueError('need covariance of parameters for computing '
                         'F statistics')

    if scale is None:
        scale = np.ones(len(params))

    F = []
    for ps, the_cov_p, s in zip(params, cov_p, scale):

        Rbq = np.dot(r_matrix, ps[:, None])

        cov = np.dot(r_matrix, np.dot(the_cov_p * s, r_matrix.T)) 
        invcov = np.linalg.inv(cov)
        J = float(r_matrix.shape[0])  # number of restrictions
        F.append(np.dot(np.dot(Rbq.T, invcov), Rbq) / J)
    return np.array(F)

class VectorizedGLM(object):

    def estimate_scale(self, mu):
        """
        Estimates the dispersion/scale.

        Type of scale can be chose in the fit method.

        Parameters
        ----------
        mu : array
            mu is the mean response estimate

        Returns
        -------
        Estimate of scale

        Notes
        -----
        The default scale for Binomial and Poisson families is 1.  The default
        for the other families is Pearson's Chi-Square estimate.

        See also
        --------
        statsmodels.glm.fit for more information
        """

        if not self.scaletype:
            if isinstance(self.family, (fam.Binomial, fam.Poisson)):
                return np.ones(len(mu))
            else:
                resid = self.endog - mu
                res = ((np.power(resid, 2) / self.family.variance(mu)).sum(axis=-1) \
                    / self.df_resid)
                return res

        if isinstance(self.scaletype, float):
            return np.array(self.scaletype)

        if isinstance(self.scaletype, str):
            if self.scaletype.lower() == 'x2':
                resid = self.endog - mu
                return ((np.power(resid, 2) / self.family.variance(mu)).sum(axis=-1) \
                    / self.df_resid)
            elif self.scaletype.lower() == 'dev':
                return self.family.deviance(self.endog, mu)/self.df_resid
            else:
                raise ValueError("Scale %s with type %s not understood" %\
                    (self.scaletype,type(self.scaletype)))

        else:
            raise ValueError("Scale %s with type %s not understood" %\
                (self.scaletype, type(self.scaletype)))


    def __init__(self, endog, exog, family=None):
        self.endog = endog
        self.exog = np.array([ exog for x in endog ])
        self.family = family

        
        self.pinv_wexog = np.array([np.linalg.pinv(foo) for foo in self.exog])
        self.df_model = rank(self.exog[0])-1
        self.df_resid = self.exog.shape[1] - rank(self.exog[0])

    def fit(self, maxiter=100, method='IRLS', tol=1e-8, scale=None):
        '''
        Fits a generalized linear model for a given family.

        parameters
        ----------
        maxiter : int, optional
            Default is 100.
        method : string
            Default is 'IRLS' for iteratively reweighted least squares.  This
            is currently the only method available for GLM fit.
        scale : string or float, optional
            `scale` can be 'X2', 'dev', or a float
            The default value is None, which uses `X2` for Gamma, Gaussian,
            and Inverse Gaussian.
            `X2` is Pearson's chi-square divided by `df_resid`.
            The default is 1 for the Binomial and Poisson families.
            `dev` is the deviance divided by df_resid
        tol : float
            Convergence tolerance.  Default is 1e-8.
        '''
        endog = self.endog
        data_weights = np.ones(endog.shape)
        self.data_weights = data_weights

        self.scaletype = scale

        mu = self.family.starting_mu(self.endog)
        wlsexog = self.exog
        eta = self.family.predict(mu)

        dev = self.family.deviance(self.endog, mu)

#        for x in dev:
#            if np.isnan(x):
#                raise ValueError("The first guess on the deviance function "
#                                 "returned a nan.  This could be a boundary "
#                                 " problem and should be reported.")

        # first guess on the deviance is assumed to be scaled by 1.
        # params are none to start, so they line up with the deviance
        history = dict(params = [None, None], deviance=[np.inf,dev])
        iteration = 0
        converged = 0
        criterion = history['deviance']
        while not converged:
            self.weights = data_weights*self.family.weights(mu)

            wlsendog = eta + self.family.link.deriv(mu) * (self.endog-mu)

            wls = VectorizedWLS(wlsendog, wlsexog, self.weights)
            (beta, normalized_cov_params) = wls.fit()

            eta = np.zeros(np.shape(self.endog))

            for i in range(len(eta)):
                eta[i] = np.dot(self.exog[i], beta[i])
            mu = self.family.fitted(eta)

            history = self._update_history(beta, mu, history)
            
            self.scale = self.estimate_scale(mu)
            iteration += 1
            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)
            converged = _check_convergence(criterion, iteration, tol,
                                            maxiter)
        self.mu = mu

        history['iteration'] = iteration

        return (beta, self.mu, self.weights, normalized_cov_params)


    def _update_history(self, beta, mu, history):
        """
        Helper method to update history during iterative fit.
        """
        history['params'].append(beta)

        dev = self.family.deviance(self.endog, mu)
        history['deviance'].append(dev)
        return history

# TODO: I think I need to fix this.
def _check_convergence(criterion, iteration, tol, maxiter):
    
    delta = np.fabs(criterion[iteration] - criterion[iteration-1])

    return np.all(delta <= tol) or iteration > maxiter

def whiten(weights, X):
    """
    Whitener for WLS model, multiplies each column by sqrt(self.weights)
    
    Parameters
    ----------
    X : array-like
    Data to be whitened
    
    Returns
    -------
    sqrt(weights)*X
    """
        #print weights.var()
    X = np.asarray(X)
    if weights.ndim == X.ndim:
        return np.sqrt(weights) * X
    elif weights.ndim + 1 == X.ndim:
        return np.sqrt(weights)[..., None] * X
    else:
        raise Exception("Incompatible shapes" + str(np.shape(weights))
                        + str(np.shape(X)))





class VectorizedWLS():

#FIXME: bug in fvalue or f_test for this example?
#UPDATE the bug is in fvalue, f_test is correct vs. R
#mse_model is calculated incorrectly according to R
#same fixed used for WLS in the tests doesn't work
#mse_resid is good
    def __init__(self, endog, exog, weights=1.):
        self.weights = weights
        self.endog = endog
        self.exog = exog
        self.wexog = whiten(weights, exog)
        self.wendog = whiten(weights, endog)
        self.params = []


    def fit(self, method="pinv", **kwargs):
        """
        Full fit of the model.

        The results include an estimate of covariance matrix, (whitened)
        residuals and an estimate of scale.

        Parameters
        ----------
        method : str
            Can be "pinv", "qr", or "mle".  "pinv" uses the
            Moore-Penrose pseudoinverse to solve the least squares problem.
            "svd" uses the Singular Value Decomposition.  "qr" uses the
            QR factorization.  "mle" fits the model via maximum likelihood.
            "mle" is not yet implemented.

        Returns
        -------
        A RegressionResults class instance.

        See Also
        ---------
        regression.RegressionResults

        Notes
        -----
        Currently it is assumed that all models will have an intercept /
        constant in the design matrix for postestimation statistics.

        The fit method uses the pseudoinverse of the design/exogenous variables
        to solve the least squares minimization.

        """
        exog = self.wexog
        endog = self.wendog

        if method == "pinv":
            if ((not hasattr(self, 'pinv_wexog')) or
                (not hasattr(self, 'normalized_cov_params'))):
                #print "recalculating pinv"   #for debugging

                inv_shape = (np.size(self.wexog, 0),
                             np.size(self.wexog, 2),
                             np.size(self.wexog, 1))

                pinv_wexog = np.zeros(inv_shape)

                normalized_cov_params = np.zeros(
                    (np.size(self.wexog, 0),
                     np.size(self.wexog, 2),
                     np.size(self.wexog, 2)))

                beta = np.zeros(np.shape(pinv_wexog)[:2])
                     
                for i in range(len(self.wexog)):
                    pinv_wexog[i] = np.linalg.pinv(self.wexog[i])
                    normalized_cov_params[i] = np.dot(pinv_wexog[i],
                                                   np.transpose(pinv_wexog[i]))
                    beta[i] = np.dot(pinv_wexog[i], endog[i])
                self.pinv_wexog = pinv_wexog


        elif method == "qr":
            if ((not hasattr(self, '_exog_Q')) or
                (not hasattr(self, 'normalized_cov_params'))):
                Q, R = np.linalg.qr(exog)
                self._exog_Q, self._exog_R = Q, R
                normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
            else:
                Q, R = self._exog_Q, self._exog_R

            beta = np.linalg.solve(R,np.dot(Q.T,endog))

        return (beta, normalized_cov_params)


if __name__ == '__main__':
    main()




