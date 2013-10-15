from __future__ import print_function

import pade.family as fam
import numpy as np
from scipy.linalg import svdvals
from collections import namedtuple

GlmResults = namedtuple('GlmResults', ['beta', 'mu', 'weights', 'normalized_cov_params', 'scale'])

def f_test_two_cond(betas, cov_ps, smoothing=0.0):
    """Special case of the f-test for when r_matrix is [ [ 0, 1 ] ].

    """
    numer = betas[:, 1] ** 2

    if np.shape(smoothing) is ():
        F = numer / (cov_ps[:, 1, 1] + smoothing)

    else:
        F = np.zeros(np.shape(smoothing) + (len(betas),))
        for j, a in enumerate(smoothing):
            F[j] = numer / (cov_ps[:, 1, 1] + a)

    return F.reshape(F.shape + (1, 1))


def f_test(betas, r_matrix, cov_ps, scale, smoothing=0.0):

    if (cov_ps is None):
        raise ValueError('need covariance of parameters for computing '
                         'F statistics')

    if scale is None:
        scale = np.ones(len(betas))

    F = None
    i = 0

    scale = np.reshape(scale, (len(scale), 1, 1))
    scaled_cov_ps = cov_ps * scale

    if np.shape(r_matrix) == (1, 2) and r_matrix[0, 0] == 0 and r_matrix[0, 1] == 1:
        return f_test_two_cond(betas, scaled_cov_ps, smoothing)

    for beta, cov_p in zip(betas, scaled_cov_ps):
        
        Rbq = np.dot(r_matrix, beta[:, None])
        
        if np.shape(smoothing) is ():
            cov = r_matrix.dot((cov_p + smoothing).dot(r_matrix.T))

            invcov = np.linalg.inv(cov)

            res = np.dot(np.dot(Rbq.T, invcov), Rbq)

            if F is None:
                F = np.zeros((len(betas),) + res.shape)
            F[i] = res

        else:
            for j, a in enumerate(smoothing):
                cov = r_matrix.dot((cov_p + a).dot(r_matrix.T))

                invcov = np.linalg.inv(cov)

                res = np.dot(np.dot(Rbq.T, invcov), Rbq)

                if F is None:
                    F = np.zeros((len(smoothing), len(betas),) + res.shape)
                F[j, i] = res

        i += 1

    J = float(r_matrix.shape[0])  # number of restrictions
    return F / J

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
        elif (scaletype.lower() == 'shrinkage') and isinstance(family, fam.NegativeBinomial):
            raise ValueError("Scale %s here" %scaletype)
            #return shrinkage_dispersion( FIXME parameters) 
        else:
            raise ValueError("Scale %s with type %s not understood" %\
                (scaletype,type(scaletype)))

    else:
        raise ValueError("Scale %s with type %s not understood" %\
            (scaletype, type(scaletype)))


def rank(X, cond=1.0e-12):
    X = np.asarray(X)
    if len(X.shape) == 2:
        D = svdvals(X)
        return int(np.add.reduce(np.greater(D / D.max(), cond).astype(np.int32)))
    else:
        return int(not np.alltrue(np.equal(X, 0.)))



def fit_glm(endog, exog, family=None, maxiter=100, tol=1e-8, scaletype=None):
    '''
    Fits a generalized linear model for a given family.

    :param endog:

    :param exog:

    :param family:

    :param maxiter:

    :param scale:

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

    return GlmResults(beta, mu, weights, normalized_cov_params, scale)


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

    if weights.ndim == X.ndim:
        return np.sqrt(weights) * X
    elif weights.ndim + 1 == X.ndim:
        return np.sqrt(weights)[..., None] * X
    else:
        raise Exception("Incompatible shapes" + str(np.shape(weights))
                        + str(np.shape(X)))


def fit_wls(endog, exog, weights=1.):
    """
    Full fit of the model.

    :return: 
      a 2-tuple with the parameters and the estimated covariance
      matrix.

    :param endog:
      the 2-d endogenous matrix

    :param exog:
      the 2-d exogenous matrix

    """
    
    wexog  = whiten(weights, exog)
    wendog = whiten(weights, endog)

    (n_models, n_obs, n_regressors) = wexog.shape
    normalized_cov_params = np.zeros((n_models, n_regressors, n_regressors))
    beta = np.zeros((n_models, n_regressors))

    for i in range(len(wexog)):
        pinv_wexog = np.linalg.pinv(wexog[i])
        normalized_cov_params[i] = np.dot(pinv_wexog, pinv_wexog.T)
        beta[i] = np.dot(pinv_wexog, wendog[i])

    return (beta, normalized_cov_params)


if __name__ == '__main__':
    main()

