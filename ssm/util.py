from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.linalg import block_diag
from autograd import grad

from scipy.optimize import linear_sum_assignment, minimize


def compute_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0
    
    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape
    assert K1 <= K2, "Can only find permutation from more states to fewer"
    
    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"
    
    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm


def random_rotation(n, theta=None):
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n, n))
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)


def ensure_args_are_lists(f):
    def wrapper(self, datas, inputs=None, masks=None, tags=None, **kwargs):
        datas = [datas] if not isinstance(datas, (list, tuple)) else datas
        
        if inputs is None:
            inputs = [np.zeros((data.shape[0], 0)) for data in datas]
        elif not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, (list, tuple)):
            masks = [masks]

        if tags is None:
            tags = [None] * len(datas)
        elif not isinstance(tags, (list, tuple)):
            tags = [tags]

        return f(self, datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

    return wrapper


def ensure_variational_args_are_lists(f):
    def wrapper(self, arg0, datas, inputs=None, masks=None, tags=None, **kwargs):
        datas = [datas] if not isinstance(datas, (list, tuple)) else datas
        
        if inputs is None:
            inputs = [np.zeros((data.shape[0], 0)) for data in datas]
        elif not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, (list, tuple)):
            masks = [masks]

        if tags is None:
            tags = [None] * len(datas)
        elif not isinstance(tags, (list, tuple)):
            tags = [tags]

        return f(self, arg0, datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

    return wrapper


def ensure_args_not_none(f):
    def wrapper(self, data, input=None, mask=None, tag=None, **kwargs):
        assert data is not None
        input = np.zeros((data.shape[0], 0)) if input is None else input
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return f(self, data, input=input, mask=mask, tag=tag, **kwargs)
    return wrapper


def ensure_slds_args_not_none(f):
    def wrapper(self, variational_mean, data, input=None, mask=None, tag=None, **kwargs):
        assert variational_mean is not None
        assert data is not None
        input = np.zeros((data.shape[0], 0)) if input is None else input
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return f(self, variational_mean, data, input=input, mask=mask, tag=tag, **kwargs)
    return wrapper


def logistic(x):
    return 1. / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def softplus(x):
    return np.log(1 + np.exp(x))


def inv_softplus(y):
    return np.log(np.exp(y) - 1)


def one_hot(z, K):
    z = np.atleast_1d(z).astype(int)
    assert np.all(z >= 0) and np.all(z < K)
    shp = z.shape
    N = z.size
    zoh = np.zeros((N, K))
    zoh[np.arange(N), np.arange(K)[np.ravel(z)]] = 1
    zoh = np.reshape(zoh, shp + (K,))
    return zoh


def relu(x):
    return np.maximum(0, x)

def batch_mahalanobis(L, x):
    """
    Copied from PyTorch torch.distributions.multivariate_normal

    Computes the squared Mahalanobis distance 
    :math:`\mathbf{x}^T \mathbf{M}^{-1}\mathbf{x}`
    for a factored 
    :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^T`.

    Accepts batches for both L and x.
    """
    # Flatten the Cholesky into a (K, D, D) array
    flat_L = np.reshape(L[None, ...], (-1,) + L.shape[-2:])
    # Invert each of the K arrays and reshape like L
    L_inv = np.reshape(np.array([np.linalg.inv(Li.T) for Li in flat_L]), L.shape)
    # Reshape x into (..., D, 1); dot with L_inv^T; square and sum.
    return np.sum(np.sum(x[..., None] * L_inv, axis=-2)**2, axis=-1)

def generalized_newton_studentst_dof(E_tau, E_logtau, nu0=1, max_iter=100, nu_min=1e-3, nu_max=20, tol=1e-8, verbose=False):
    """
    Generalized Newton's method for the degrees of freedom parameter, nu, 
    of a Student's t distribution.  See the notebook in the doc/students_t
    folder for a complete derivation. 
    """
    from scipy.special import digamma, polygamma
    delbo = lambda nu: 1/2 * (1 + np.log(nu/2)) - 1/2 * digamma(nu/2) + 1/2 * E_logtau - 1/2 * E_tau
    ddelbo = lambda nu: 1/(2 * nu) - 1/4 * polygamma(1, nu/2)

    dnu = np.inf
    nu = nu0
    for itr in range(max_iter):
        if abs(dnu) < tol:
            break
            
        if nu < nu_min or nu > nu_max:
            warn("generalized_newton_studentst_dof fixed point grew beyond "
                 "bounds [{},{}].".format(nu_min, nu_max))
            nu = np.clip(nu, nu_min, nu_max)
            break
        
        # Perform the generalized Newton update
        a = -nu**2 * ddelbo(nu)
        b = delbo(nu) - a / nu
        assert a > 0 and b < 0, "generalized_newton_studentst_dof encountered invalid values of a,b"
        dnu = -a / b - nu
        nu = nu + dnu

    if itr == max_iter - 1:
        warn("generalized_newton_studentst_dof failed to converge" 
             "at tolerance {} in {} iterations.".format(tol, itr))

    return nu

def fit_multiclass_logistic_regression(X, y, bias=None, K=None, W0=None, mu0=0, sigmasq0=1, 
                                       solver="lbfgs", verbose=False, maxiter=1000):
    """
    Fit a multiclass logistic regression 

        y_i ~ Cat(softmax(W x_i))

    y is a one hot vector in {0, 1}^K  
    x_i is a vector in R^D
    W is a matrix R^{K x D}

    The log likelihood is,

        L(W) = sum_i sum_k y_ik * w_k^T x_i - logsumexp(W x_i)
    
    The prior is w_k ~ Norm(mu0, diag(sigmasq0)).
    """
    N, D = X.shape
    assert y.shape[0] == N

    # Make sure y is one hot
    if y.ndim == 1 or y.shape[1] == 1:
        assert y.dtype == int and y.min() >= 0
        K = y.max() + 1 if K is None else K
        y_oh = np.zeros((N, K), dtype=int)
        y_oh[np.arange(N), y] = 1

    else:
        K = y.shape[1]
        assert y.min() == 0 and y.max() == 1 and np.allclose(y.sum(1), 1)
        y_oh = y

    # Check that bias is correct shape
    if bias is not None:
        assert bias.shape == (K,) or bias.shape == (N, K)
    else:
        bias = np.zeros((K,))

    def loss(W_flat):
        W = np.reshape(W_flat, (K, D))
        scores = np.dot(X, W.T) + bias
        lp = np.sum(y_oh * scores) - np.sum(logsumexp(scores, axis=1))
        prior = np.sum(-0.5 * (W - mu0)**2 / sigmasq0)
        return -(lp + prior) / N

    W0 = W0 if W0 is not None else np.zeros((K, D))
    assert W0.shape == (K, D)

    itr = [0]
    def callback(W_flat):
        itr[0] += 1
        print("Iteration {} loss: {:.3f}".format(itr[0], loss(W_flat)))

    result = minimize(loss, np.ravel(W0), jac=grad(loss), 
                      method="BFGS", 
                      callback=callback if verbose else None, 
                      options=dict(maxiter=maxiter, disp=verbose))

    W = np.reshape(result.x, (K, D))
    return W

def fit_linear_regression(Xs, ys, weights=None, 
                          mu0=0, sigmasq0=1, alpha0=1, beta0=1, 
                          fit_intercept=True):
    """
    Fit a linear regression y_i ~ N(Wx_i + b, diag(S)) for W, b, S.
    
    :param Xs: array or list of arrays
    :param ys: array or list of arrays
    :param fit_intercept:  if False drop b
    """
    Xs = Xs if isinstance(Xs, (list, tuple)) else [Xs]
    ys = Xs if isinstance(ys, (list, tuple)) else [ys]
    assert len(Xs) == len(ys)

    D = Xs[0].shape[1]
    P = ys[0].shape[1]
    assert all([X.shape[1] == D for X in Xs])
    assert all([y.shape[1] == P for y in ys])
    assert all([X.shape[0] == y.shape[0] for X, y in zip(Xs, ys)])
    
    mu0 = mu0 * np.zeros((P, D))
    sigmasq0 = sigmasq0 * np.eye(D)

    if weights is None:
        weights = [np.ones(X.shape[0]) for X in Xs]

    # Add weak prior on intercept
    if fit_intercept:
        mu0 = np.column_stack((mu0, np.zeros(P)))
        sigmasq0 = block_diag(sigmasq0, np.eye(10))

    # Compute the posterior
    J = np.linalg.inv(sigmasq0)
    h = np.dot(J, mu0.T)

    for X, y in zip(Xs, ys):
        X = np.column_stack((X, np.ones(X.shape[0]))) if fit_intercept else X
        J += np.dot(X.T * weights, X)
        h += np.dot(X.T * weights, y)
    
    # Solve for the MAP estimate    
    W = np.linalg.solve(J, h).T

    # Compute the residual and the posterior variance
    alpha = alpha0 
    beta = beta0 * np.ones(P)
    for X, y in zip(Xs, ys):
        resid = y - np.dot(X, W.T)
        alpha += 0.5 * np.sum(weights)
        beta += 0.5 * np.sum(weights[:, None] * resid**2, axis=0)

    # Get MAP estimate of posterior mode of precision
    sigmasq = beta / (alpha + 1e-16)

    if fit_intercept:
        return W[:,:-1], W[:,-1], sigmasq
    else:
        return W, sigmasq
        




