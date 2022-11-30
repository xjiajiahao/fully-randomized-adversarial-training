import numpy as np
from comm import *


class naive_problem:
    def __init__(self, n_samples_train, n_samples_test, n_classifiers_candidates, n_classifiers, epsilon,
                 nb_iter=10000, lr=2., lr_position=2., iter_models=40, iter_weights=10, seed=12):
        self.seed = seed
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.n_classifiers_candidates = n_classifiers_candidates
        self.n_classifiers = n_classifiers
        self.epsilon = epsilon
        self.nb_iter = nb_iter
        self.lr = lr
        self.lr_position = lr_position
        self.iter_models = iter_models
        self.iter_weights = iter_weights

    def __call__(self):
        np.random.seed(12)
        X, Y = simulate_data(1 / 2, self.n_samples_train)
        X_out, Y_out = simulate_data(1 / 2, self.n_samples_test)
        u = simulate_uniform(1000)  # random perbutations

        np.random.seed(self.seed)
        ws, bs = classifiers(self.n_classifiers, symmetrized=False)

        lbda = np.ones(self.n_classifiers) / self.n_classifiers  # the weights of the classifers, subject to the simplex

        losses_in = []

        count_model_updates = 0
        count_weight_updates = 0
        tmp_ws = np.zeros([1, ws.shape[1]])
        tmp_bs = np.zeros(1)
        tmp_lbda = np.ones(1)
        for i in range(self.nb_iter):
            if i % (self.iter_models + self.iter_weights) < self.iter_models:
                # sample a model to update
                idx_to_update = np.random.randint(0, self.n_classifiers)
                tmp_ws[0] = ws[idx_to_update]
                tmp_bs[0] = bs[idx_to_update]
                perturbed_X = generate_perturbed_samples(tmp_ws, tmp_bs, tmp_lbda, X, Y, self.epsilon)
                adv_loss, grad_ws, grad_bs = compute_perturbed_loss(
                    perturbed_X, Y, tmp_ws, tmp_bs)
                # losses_in.append(adv_loss)  # save the losses

                lr = self.lr_position / (count_model_updates // self.n_classifiers + 1) ** 0.5
                tmp_ws = tmp_ws - lr * grad_ws
                tmp_bs = tmp_bs - lr * grad_bs
                ws[idx_to_update] = tmp_ws[0]
                bs[idx_to_update] = tmp_bs[0]
                count_model_updates += 1
            else:
                # update the weights
                adv_loss, grad_avg = compute_mixed_classifer_logistic_risk_adv(X, Y, lbda, u, ws, bs, self.epsilon)
                losses_in.append(adv_loss)  # save the losses

                lbda -= self.lr / (count_weight_updates + 1) ** 0.5 * (grad_avg)  # projected gradient descent
                count_weight_updates += 1
                lbda = projection_simplex_sort(lbda)
                losses_in.append(adv_loss)  # save the losses

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
        output_file = './data/results_atm_seed_{:02d}.csv'.format(seed)
        results = np.zeros((51, 9))
        for i in range(51):
            eps = i * 0.1
            problem = naive_problem(100, 100, 200, 20, eps, nb_iter=2000, lr=1., lr_position=1., iter_models=40, iter_weights=10, seed=seed)
            curr_result = problem()
            data_to_log = np.array([[eps] + curr_result])
            results[i] = np.array([eps] + curr_result)
            i += 1
            with open(output_file, 'a') as f:
                np.savetxt(f, data_to_log, delimiter=', ')
