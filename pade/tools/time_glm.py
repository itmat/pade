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

    (params, mu, weights, cov_p, scale) = fit_glm(y, x, family)
    f = f_test(params, contrast, cov_p, scale)

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

    y = np.genfromtxt('data.txt')

    x = np.zeros((24, 2), int)
    x[:, 0] = 1
    x[12:, 1] = 1

    contrast = np.array([ [0, 1] ])

    import statsmodels.api as sm


#    time_glm(y, x, sm.families.Gamma(), fam.Gamma(), contrast)

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

def estimate_scale(mu, family, endog, scaletype=None, df_resid=None):
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

    if not scaletype:
        if isinstance(family, (fam.Binomial, fam.Poisson)):
            return np.ones(len(mu))
        else:
            resid = endog - mu
            res = ((np.power(resid, 2) / family.variance(mu)).sum(axis=-1) \
                / df_resid)
            return res

    if isinstance(scaletype, float):
        return np.array(scaletype)

    if isinstance(scaletype, str):
        if scaletype.lower() == 'x2':
            resid = endog - mu
            return ((np.power(resid, 2) / family.variance(mu)).sum(axis=-1) \
                / df_resid)
        elif scaletype.lower() == 'dev':
            return family.deviance(endog, mu)/df_resid
        else:
            raise ValueError("Scale %s with type %s not understood" %\
                (scaletype,type(scaletype)))

    else:
        raise ValueError("Scale %s with type %s not understood" %\
            (scaletype, type(scaletype)))


def fit_glm(endog, exog, family=None, maxiter=100, tol=1e-8, scaletype=None):
    '''
    Fits a generalized linear model for a given family.

    parameters
    ----------
    maxiter : int, optional
        Default is 100.
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

    exog = np.array([ exog for x in endog ])

    mu  = family.starting_mu(endog)
    eta = family.predict(mu)
    dev = family.deviance(endog, mu)

    for x in dev:
        if np.isnan(x):
            raise ValueError("The first guess on the deviance function "
                             "returned a nan.  This could be a boundary "
                             " problem and should be reported.")

    iteration = 0
    converged = 0

    deviance = [ np.inf, dev ]

    df_resid = exog.shape[1] - rank(exog[0])

    while not converged:
        weights  = family.weights(mu)
        wlsendog = eta + family.link.deriv(mu) * (endog - mu)

        (beta, normalized_cov_params) = fit_wls(wlsendog, exog, weights)

        eta = np.zeros(np.shape(endog))

        for i in range(len(eta)):
            eta[i] = np.dot(exog[i], beta[i])
        mu = family.fitted(eta)

        deviance.append(family.deviance(endog, mu))

        scale = estimate_scale(mu, family=family, endog=endog, scaletype=scaletype, df_resid=df_resid)
        iteration += 1
        if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
            msg = "Perfect separation detected, results not available"
            raise PerfectSeparationError(msg)
        converged = _check_convergence(deviance, iteration, tol, maxiter)

    return (beta, mu, weights, normalized_cov_params, scale)


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


def fit_wls(endog, exog, weights=1., method="pinv", **kwargs):
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

    wexog  = whiten(weights, exog)
    wendog = whiten(weights, endog)

    if method == "pinv":

        inv_shape = (np.size(wexog, 0),
                     np.size(wexog, 2),
                     np.size(wexog, 1))

        pinv_wexog = np.zeros(inv_shape)

        normalized_cov_params = np.zeros(
            (np.size(wexog, 0),
             np.size(wexog, 2),
             np.size(wexog, 2)))

        beta = np.zeros(np.shape(pinv_wexog)[:2])

        for i in range(len(wexog)):
            pinv_wexog[i] = np.linalg.pinv(wexog[i])
            normalized_cov_params[i] = np.dot(pinv_wexog[i],
                                           np.transpose(pinv_wexog[i]))
            beta[i] = np.dot(pinv_wexog[i], wendog[i])

    elif method == "qr":
        Q, R = np.linalg.qr(exog)
        self._exog_Q, self._exog_R = Q, R
        normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
        beta = np.linalg.solve(R,np.dot(Q.T,wendog))

    return (beta, normalized_cov_params)


if __name__ == '__main__':
    main()




