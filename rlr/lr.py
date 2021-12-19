import numpy as np
import lbfgs
from .crossvalidation import gridSearch

class RegularizedLogisticRegression(object) :
    def __init__(self, alpha=0, cv=3, num_cores=0) :
        self.alpha = alpha
        self.cv = cv
        self.num_cores = num_cores

    def fit(self, examples, labels, case_weights=None, cv=True) :
        if cv and self.cv :
            self.alpha = gridSearch(examples, 
                                    labels, 
                                    self, 
                                    self.num_cores, 
                                    self.cv)

        return self.fit_alpha(examples, labels, case_weights)

    def fit_alpha(self, examples, labels, case_weights) :

        if case_weights is None :
            case_weights = np.ones(examples.shape[0])

        # {0,1} -> {-1,1}
        labels = labels * 2 - 1

        start_betas = np.ones(examples.shape[1] + 1)

        opt = lbfgs.LBFGS()
        opt.epsilon = 0.000001
        opt.linesearch = 'strongwolfe'

        final_betas = opt.minimize(loss,
                                   x0 = start_betas,
                                   progress=None,
                                   args = (examples, labels, 
                                           case_weights, self.alpha)) 


        self.weights = final_betas[:examples.shape[1]]
        self.bias = final_betas[-1] 

    def predict_proba(self, examples) :
        scores = np.dot(examples, self.weights)
        scores = np.exp(scores + self.bias) / (1 + np.exp(scores + self.bias))
        
        return scores.reshape(-1, 1)

def phi(t):
    # logistic function, returns 1 / (1 + exp(-t))
    idx = t > 0
    out = np.empty(t.size, dtype=float)
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
    out = out.sum() + .5 * alpha * w.dot(w)
    
    g[:] = gradient(x0, X, y, case_weights, alpha)

    return out



def gradient(x0, X, y, case_weights, alpha):
    # gradient of the logistic loss
    w, c = x0[:X.shape[1]], x0[-1]
    z = X.dot(w) + c
    z = phi(y * z) 
    z0 = (z - 1) * y * case_weights
    grad_w = X.T.dot(z0) + alpha * w
    grad_c = z0.sum() 
    return np.concatenate((grad_w, [grad_c]))



def progress(x, g, fx, xnorm, gnorm, step, k, ls, *args):
    print('fx', fx)
    print('gnorm', gnorm)
    print('iteration', k)

    return 0
