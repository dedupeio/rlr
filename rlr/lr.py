import numpy as np
import lbfgs

def phi(t):
    # logistic function, returns 1 / (1 + exp(-t))
    idx = t > 0
    out = np.empty(t.size, dtype=np.float)
    out[idx] = 1. / (1. + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out


def loss(x0, g, X, y, case_weights, alpha):
    # logistic loss function, returns Sum{-log(phi(t))}
    w, c = x0[:X.shape[1]], x0[-1]
    z = X.dot(w) + c
    yz = y * z
    idx = yz > 0
    out = np.zeros_like(yz)
    out[idx] = np.log(1 + np.exp(-yz[idx]))
    out[~idx] = (-yz[~idx] + np.log(1. + np.exp(yz[~idx])))
    out *= case_weights
    out = out.sum() / X.shape[0] + .5 * alpha * w.dot(w)
    
    g[:] = gradient(x0, X, y, case_weights, alpha)

    return out



def gradient(x0, X, y, case_weights, alpha):
    # gradient of the logistic loss
    w, c = x0[:X.shape[1]], x0[-1]
    z = X.dot(w) + c
    z = phi(y * z) 
    z0 = (z - 1) * y * case_weights
    grad_w = X.T.dot(z0) / X.shape[0] + alpha * w
    grad_c = z0.sum() / X.shape[0]
    return np.concatenate((grad_w, [grad_c]))


def lr(labels, examples, alpha, case_weights = None) :

    if case_weights is None :
        case_weights = np.ones(examples.shape[0])

    # {0,1} -> {-1,1}
    labels = labels * 2 - 1

    start_betas = np.ones(examples.shape[1] + 1)

    opt = lbfgs.LBFGS()
    opt.xtol = .00000000000000000000001

    final_betas = opt.minimize(loss,
                               x0 = start_betas,
                               progress=None,
                               args = (examples, labels, 
                                       case_weights, alpha)) 

    
    return final_betas[:examples.shape[1]], final_betas[-1] 
