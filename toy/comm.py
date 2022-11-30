import numpy as np
from scipy.special import softmax, logsumexp, expit
import math

mean_0 = [0, 0]  # P(X | Y = -1) = N([0, 0], I_2)
mean_10 = [3, 0]  # P(X | Y = 1) = (1/4) N([-3, 0], I_2) + (3/4) N([3, 0], I_2)
mean_11 = [-3, 0]
# mean_11 = [3, 0]
cov = np.eye(2)
x_lim = (-7, 7)
y_lim = (-7, 7)


def simulate_uniform(n):  # draw n samples uniformly from the epsilon-ball
    rho = np.sqrt(np.random.uniform(size=n))
    theta = 2 * np.pi * np.random.uniform(size=n)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    a = np.zeros((n, 2))
    a[:, 0] = x
    a[:, 1] = y
    return a


def compute_linear_classifier_01_risk_adv(w, b, X, y, epsilon):
    return np.mean(((X.dot(w.T) + b).T * y).T <= +epsilon * np.linalg.norm(w, axis=1), axis=0)  # the worst perturbation is -y * w / norm(w)



def simulate_data(p, n):
    n_0 = np.random.binomial(n, p) # the number of negative samples
    n_1 = n - n_0 # the number of postive samples
    Y = np.ones(n) # labels
    Y[:n_0] = -1

    data_0 = np.random.multivariate_normal(mean_0, cov, n_0)  # genrate features of positive samples

    # n_10 = np.random.binomial(n_1, 1 / 2)  # number of positive samples from the first component
    n_10 = np.random.binomial(n_1, 3 / 4)  # number of positive samples from the first component
    n_11 = n_1 - n_10  # number of positive samples from the second component
    data_10 = np.random.multivariate_normal(mean_10, cov, n_10)  # generate positive samples from the first component
    data_11 = np.random.multivariate_normal(mean_11, cov, n_11)  # generate positive samples from the second component
    data_1 = np.concatenate([data_10, data_11])

    X = np.concatenate([data_0, data_1])
    return X, Y


def classifiers(num, symmetrized=True):  # generate random classifers
    x1s = np.random.uniform(x_lim[0], x_lim[1], num)  # (x1, y1) and (x2, y2) determines one linear classifier of the form w.^T (x, y) + b = 0
    y1s = np.random.uniform(y_lim[0], y_lim[1], num)

    x2s = np.random.uniform(x_lim[0], x_lim[1], num)
    y2s = np.random.uniform(y_lim[0], y_lim[1], num)

    ds = np.zeros((num, 2))
    ds[:, 0] = x2s - x1s
    ds[:, 1] = y2s - y1s
    ws = np.zeros(np.shape(ds))
    ws[:, 0] = -ds[:, 1]
    ws[:, 1] = ds[:, 0]
    bs = -(ws[:, 0] * x1s + ws[:, 1] * y1s)
    if symmetrized:
        ws = np.concatenate([ws, -ws])  # the original classifer and the symmetry one
        bs = np.concatenate([bs, -bs])
    return ws, bs


def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def compute_sup(lbda, u, ws, bs, x, y, epsilon):
    X = x + epsilon * u  # x: 1-by-2, u: num_random_perturbations-by-2
    z = ((np.matmul(X, ws.T) + bs).T * y).T  # num_random_perturbations-by-n_classifiers
    sign_z = np.sign(z)
    loss = np.log(1 + np.exp(-np.abs(z))) + (sign_z - 1) / 2 * z
    mixed_loss = loss.dot(lbda)
    adv_index = np.argmax(mixed_loss)  # the worst average loss
    return mixed_loss[adv_index], loss[adv_index]


def compute_sup_regularized(lbda, u, ws, bs, x, y, epsilon, eta):
    X = x + epsilon * u
    z = ((np.matmul(X, ws.T) + bs).T * y).T  # num_random_perturbations-by-n_classifiers
    sign_z = np.sign(z)
    loss = np.log(1 + np.exp(-np.abs(z))) + (sign_z - 1) / 2 * z
    mixed_loss = loss.dot(lbda)
    softmaxes = softmax(mixed_loss / eta)  # softmax with regularization coefficient eta
    grad = np.dot(softmaxes, loss / eta)  # gradient w.r.t. lbda

    return eta * logsumexp(mixed_loss / eta), grad


def compute_mixed_classifer_logistic_risk_adv(X, Y, lbda, u, ws, bs, epsilon, reg=-1):
    n_samples = len(X)
    if epsilon == 0:
        u = np.zeros((1, 2))
    global_adv_loss = 0
    grads = []
    for x, y in zip(X, Y):
        if reg < 0:  # compute the unregularized adversarial risk of the randomized classifier
            adv_loss, grad_wrt_lbda = compute_sup(
                lbda, u, ws, bs, x, y, epsilon)
        else:  # compute the regularized adversarial risk of the randomized classifier (for Meunier's algorithm)
            adv_loss, grad_wrt_lbda = compute_sup_regularized(
                lbda, u, ws, bs, x, y, epsilon, reg)

        global_adv_loss += adv_loss
        grads.append(grad_wrt_lbda)
    global_adv_loss /= n_samples
    grad_avg = np.array(grads).mean(axis=0)
    return global_adv_loss, grad_avg


def compute_classification_error_adv(X, Y, lbda, u, ws, bs, epsilon, reg=-1):
    n_samples = len(X)
    if epsilon == 0:
        u = np.zeros((1, 2))
    global_adv_loss = 0
    grads = []
    for x, y in zip(X, Y):
        if reg < 0:  # compute the unregularized adversarial risk of the randomized classifier
            perturbed_X = x + epsilon * u  # x: 1-by-2, u: num_random_perturbations-by-2
            z = ((perturbed_X.dot(ws.T) + bs).T * y).T <= 0  # 0-1 loss, num_random_perturbations-by-num_classifiers
            a = z.dot(lbda)  # average loss, num_random_perturbations-by-1
            adv_index = np.argmax(a)  # the worst average loss
            adv_loss = a[adv_index]
            grad = z[adv_index]
        else:  # compute the regularized adversarial risk of the randomized classifier
            perturbed_X = x + epsilon * u
            losses = ((perturbed_X.dot(ws.T) + bs).T * y).T <= 0
            a = losses.dot(lbda)
            softmaxes = softmax(a / reg)  # softmax with regularization coefficient eta
            grad = np.dot(softmaxes, losses / reg)  # gradient w.r.t. lbda
            adv_loss = reg * logsumexp(a / reg)

        global_adv_loss += adv_loss
        grads.append(grad)

    global_adv_loss /= n_samples
    grad_avg = np.array(grads).mean(axis=0)
    return global_adv_loss, grad_avg


def compute_perturbed_loss(perturbed_X, Y, ws, bs):
    # logistic loss
    n_samples = len(perturbed_X)  # n_samples-by-dim
    z = ((np.matmul(perturbed_X, ws.T) + bs).T * Y).T  # n_samples-by-n_classifiers
    sigmoid_z = expit(z)
    sign_z = np.sign(z)
    loss = np.mean(np.log(1 + np.exp(-np.abs(z))) + (sign_z - 1) / 2 * z, 0)  # vector of size n_classifiers  # numerically stable implementation of (-log sigmoid(z))
    # grad_ws = ((\partial loss)/(\partial z)).T \times ((X.T * Y)).T * (1.0/n_samples)
    grad_ws = np.matmul((sigmoid_z - 1.0).T, (perturbed_X.T * Y).T) / n_samples
    # grad_bs = ((\partial loss)/(\partial z)).T \times Y * (1.0/n_samples)
    grad_bs = np.matmul((sigmoid_z - 1.0).T, Y) / n_samples
    return loss, grad_ws, grad_bs

def compute_grad_X_perturbed_loss(perturbed_X, Y, ws, bs):
    # logistic loss
    n_classifiers = len(bs)  # n_samples-by-dim
    z = ((np.matmul(perturbed_X, ws.T) + bs).T * Y).T  # n_samples-by-n_classifiers
    sigmoid_z = expit(z)
    sign_z = np.sign(z)
    loss = np.mean(np.log(1 + np.exp(-np.abs(z))) + (sign_z - 1) / 2 * z, 0)  # vector of size n_classifiers  # numerically stable implementation of (-log sigmoid(z))
    grad_X = ((np.matmul(sigmoid_z - 1.0, ws)).T * (Y / n_classifiers)).T
    return loss, grad_X

def sample_from_logit_best_response(list_ws, list_bs, list_lbda, X, Y, epsilon, eta, num_iters=20, lr=0.1, is_stochastic=True, batch_size=100):
    perturbed_X = np.copy(X)
    for _ in range(num_iters):
        if is_stochastic and len(list_ws) > batch_size:
            list_idx = list(range(len(list_ws)))
            np.random.shuffle(list_idx)
            list_idx = list_idx[0:batch_size]
        else:
            list_idx = list(range(len(list_ws)))

        for j in list_idx:
            ws = list_ws[j]
            bs = list_bs[j]
            lbda = list_lbda[j]
            z = ((np.matmul(perturbed_X, ws.T) + bs).T * Y).T  # n_samples-by-n_classifiers
            sigmoid_z = expit(z)
            perturbed_X -= ((np.matmul(1.0 - sigmoid_z, (ws.T * (lr / 2 / len(list_idx) / eta * lbda)).T)).T * Y).T  # n_samples-by-dim
        perturbed_X += np.random.normal(scale=math.sqrt(lr), size=np.shape(perturbed_X))
        # project back
        scale = np.minimum(epsilon / np.sqrt(np.sum((perturbed_X - X)**2, axis=1)), 1.0)
        perturbed_X = X + ((perturbed_X - X).T * scale).T
    return perturbed_X

def generate_perturbed_samples(ws, bs, lbda, X, Y, epsilon, num_iters=20, lr=0.1):
    perturbed_X = np.copy(X)
    if epsilon != 0:
        for _ in range(num_iters):
            z = ((np.matmul(perturbed_X, ws.T) + bs).T * Y).T  # n_samples-by-n_classifiers
            sigmoid_z = expit(z)
            perturbed_X -= ((np.matmul(1.0 - sigmoid_z, (ws.T * (lr * lbda)).T)).T * Y).T  # n_samples-by-dim
            # project back
            scale = np.minimum(epsilon / np.sqrt(np.sum((perturbed_X - X)**2, axis=1)), 1.0)
            perturbed_X = X + ((perturbed_X - X).T * scale).T
    return perturbed_X
