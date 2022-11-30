import numpy as np
from comm import *


class naive_problem:
    def __init__(self, n_samples_train, n_samples_test, n_classifiers_candidates, n_classifiers, epsilon,
                 nb_iter=10000, lr=2., reg=1e-8, seed=12):
        self.seed = seed
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.n_classifiers_candidates = n_classifiers_candidates
        self.n_classifiers = n_classifiers
        self.epsilon = epsilon
        self.nb_iter = nb_iter
        self.lr = lr
        self.reg = reg

    def __call__(self):
        np.random.seed(12)
        X, Y = simulate_data(1 / 2, self.n_samples_train)
        X_out, Y_out = simulate_data(1 / 2, self.n_samples_test)
        u = simulate_uniform(1000)  # random perbutations

        np.random.seed(self.seed)
        ws, bs = classifiers(int(self.n_classifiers_candidates / 2))  # generate random classifiers

        risk_insample = compute_linear_classifier_01_risk_adv(ws, bs, X, Y, 0.)  # compute the natural risks of the classifiers
        ind_correct = risk_insample < 0.4
        ws = ws[ind_correct][:self.n_classifiers]
        bs = bs[ind_correct][:self.n_classifiers]

        adv_risk_insample = compute_linear_classifier_01_risk_adv(ws, bs, X, Y, self.epsilon)
        risk_insample = compute_linear_classifier_01_risk_adv(ws, bs, X, Y, 0.)
        opt_classifier = np.argmin(adv_risk_insample)


        lbda = np.ones(self.n_classifiers) / self.n_classifiers  # the weights of the classifers, subject to the simplex

        losses_in = []

        for i in range(self.nb_iter):
            adv_loss, grad_avg = compute_mixed_classifer_logistic_risk_adv(X, Y, lbda, u, ws, bs, self.epsilon, reg=self.reg)
            losses_in.append(adv_loss)  # save the losses

            lbda -= self.lr / (i + 1) ** 0.5 * (grad_avg)  # projected gradient descent
            lbda = projection_simplex_sort(lbda)

        losses_in = np.array(losses_in)

        risk_nat_rand_in, _ = compute_mixed_classifer_logistic_risk_adv(X, Y, lbda, u, ws, bs, 0.)
        risk_adv_rand_in, _ = compute_mixed_classifer_logistic_risk_adv(X, Y, lbda, u, ws, bs, self.epsilon)
        risk_nat_rand_out, _ = compute_mixed_classifer_logistic_risk_adv(X_out, Y_out, lbda, u, ws, bs, 0.)
        risk_adv_rand_out, _ = compute_mixed_classifer_logistic_risk_adv(X_out, Y_out, lbda, u, ws, bs, self.epsilon)

        error_nat_rand_in, _ = compute_classification_error_adv(X, Y, lbda, u, ws, bs, 0.)
        error_adv_rand_in, _ = compute_classification_error_adv(X, Y, lbda, u, ws, bs, self.epsilon)
        error_nat_rand_out, _ = compute_classification_error_adv(X_out, Y_out, lbda, u, ws, bs, 0.)
        error_adv_rand_out, _ = compute_classification_error_adv(X_out, Y_out, lbda, u, ws, bs, self.epsilon)

        return [risk_adv_rand_in, risk_nat_rand_in, risk_adv_rand_out, risk_nat_rand_out, error_adv_rand_in, error_nat_rand_in, error_adv_rand_out, error_nat_rand_out] 

if __name__ == "__main__":
    for seed in range(0, 6):
        print('current seed: {:d}'.format(seed))
        output_file = './data/results_regularized_seed_{:02d}.csv'.format(seed)
        results = np.zeros((51, 9))
        for i in range(51):
            eps = i * 0.1
            problem = naive_problem(100, 100, 200, 20, eps, nb_iter=1000, lr=0.005, reg=1e-2, seed=seed)
            curr_result = problem()
            data_to_log = np.array([[eps] + curr_result])
            results[i] = np.array([eps] + curr_result)
            i += 1
            with open(output_file, 'a') as f:
                np.savetxt(f, data_to_log, delimiter=', ')
