from __future__ import print_function

import statsmodels.genmod.families as families
import statsmodels.regression.linear_model as lm

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

    model = VectorizedGLM(y, x, family)
    fitted = model.fit()
    f = fitted.f_test(contrast)

    return GlmResults(y.T, x, family, contrast, model, fitted, f)



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

    def __init__(self, endog, exog, family=None):
        self.endog = endog
        self.exog = np.array([ exog for x in endog ])
        self.family = family

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

        if np.shape(self.data_weights) == () and self.data_weights>1:
            self.data_weights = self.data_weights *\
                    np.ones((endog.shape[0]))
        self.scaletype = scale
        if isinstance(self.family, families.Binomial):
        # this checks what kind of data is given for Binomial.
        # family will need a reference to endog if this is to be removed from
        # preprocessing
            self.endog = self.family.initialize(self.endog)

        if hasattr(self, 'offset'):
            offset = self.offset
        elif hasattr(self, 'exposure'):
            offset = self.exposure
        else:
            offset = 0
        #TODO: would there ever be both and exposure and an offset?


        mu = self.family.starting_mu(self.endog)
        wlsexog = self.exog
        eta = self.family.predict(mu)
        dev = self.family.deviance(self.endog, mu)
        if np.isnan(dev):
            raise ValueError("The first guess on the deviance function "
                             "returned a nan.  This could be a boundary "
                             " problem and should be reported.")


        # first guess on the deviance is assumed to be scaled by 1.
        # params are none to start, so they line up with the deviance
        history = dict(params = [None, None], deviance=[np.inf,dev])
        iteration = 0
        converged = 0
        criterion = history['deviance']
        while not converged:
            self.weights = data_weights*self.family.weights(mu)
            wlsendog = eta + self.family.link.deriv(mu) * (self.endog-mu) \
                - offset

            print("Calling wls with", np.shape(wlsendog),
                  np.shape(wlsexog),
                  np.shape(self.weights))

            wls = VectorizedWLS(wlsendog, wlsexog, self.weights)
            wls_results = wls.fit()

            wls_results_params = wls.beta
            print("Beta is", wls.beta)
            eta = np.dot(self.exog, wls_results_params) + offset
            mu = self.family.fitted(eta)
            history = self._update_history(wls_results, mu, history)
            self.scale = self.estimate_scale(mu)
            iteration += 1
            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)
            converged = _check_convergence(criterion, iteration, tol,
                                            maxiter)
        self.mu = mu
        glm_results = GLMResults(self, wls_results.params,
                                 wls_results.normalized_cov_params,
                                 self.scale)
        history['iteration'] = iteration
        glm_results.fit_history = history
        return GLMResultsWrapper(glm_results)


class VectorizedWLS(lm.GLS):

#FIXME: bug in fvalue or f_test for this example?
#UPDATE the bug is in fvalue, f_test is correct vs. R
#mse_model is calculated incorrectly according to R
#same fixed used for WLS in the tests doesn't work
#mse_resid is good
    def __init__(self, endog, exog, weights=1.):
        self.weights = weights
        self.endog = endog
        self.exog = exog
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        self.params = []

    def whiten(self, X):
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
        #print self.weights.var()
        X = np.asarray(X)
        if self.weights.ndim == X.ndim:
            return np.sqrt(self.weights) * X
        elif self.weights.ndim + 1 == X.ndim:
            return np.sqrt(self.weights)[..., None] * X
        else:
            raise Exception("Incompatible shapes" + str(np.shape(self.weights))
                            + str(np.shape(X)))


    def loglike(self, params):
        """
        Returns the value of the gaussian loglikelihood function at params.

        Given the whitened design matrix, the loglikelihood is evaluated
        at the parameter vector `params` for the dependent variable `Y`.

        Parameters
        ----------
        params : array-like
            The parameter estimates.

        Returns
        -------
        The value of the loglikelihood function for a WLS Model.

        Notes
        --------
        .. math:: -\\frac{n}{2}\\log\\left(Y-\\hat{Y}\\right)-\\frac{n}{2}\\left(1+\\log\\left(\\frac{2\\pi}{n}\\right)\\right)-\\frac{1}{2}log\\left(\\left|W\\right|\\right)

        where :math:`W` is a diagonal matrix
        """
        nobs2 = self.nobs / 2.0
        SSR = ss(self.wendog - np.dot(self.wexog,params))
        #SSR = ss(self.endog - np.dot(self.exog,params))
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1+np.log(np.pi/nobs2))*nobs2  # with constant
        if np.all(self.weights != 1):    #FIXME: is this a robust-enough check?
            llf -= .5*np.log(np.multiply.reduce(1/self.weights)) # with weights
        return llf

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
                self.normalized_cov_params = normalized_cov_params
                self.beta = beta


        elif method == "qr":
            if ((not hasattr(self, '_exog_Q')) or
                (not hasattr(self, 'normalized_cov_params'))):
                Q, R = np.linalg.qr(exog)
                self._exog_Q, self._exog_R = Q, R
                self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
            else:
                Q, R = self._exog_Q, self._exog_R

            beta = np.linalg.solve(R,np.dot(Q.T,endog))

            # no upper triangular solve routine in numpy/scipy?
        if isinstance(self, lm.OLS):
            lfit = OLSResults(self, beta,
                       normalized_cov_params=self.normalized_cov_params)
        else:
            lfit = lm.RegressionResults(self, beta,
                       normalized_cov_params=self.normalized_cov_params)
        return lm.RegressionResultsWrapper(lfit)


if __name__ == '__main__':
    main()




