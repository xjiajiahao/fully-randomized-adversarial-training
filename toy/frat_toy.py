import numpy as np
from comm import *


class naive_problem:
    def __init__(self, n_samples_train, n_samples_test, n_classifiers_candidates, n_classifiers, epsilon,
                 nb_iter=10000, lr_position=2., lr_weight=1., noise_level=0.01, eta=0.05, n_iters_lmc=20, lr_lmc=0.1, seed=12):
        self.seed = seed
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.n_classifiers_candidates = n_classifiers_candidates
        self.n_classifiers = n_classifiers
        self.epsilon = epsilon
        self.nb_iter = nb_iter
        self.lr_position = lr_position
        self.lr_weight = lr_weight
        self.noise_level = noise_level
        self.eta = eta
        self.n_iters_lmc = n_iters_lmc
        self.lr_lmc = lr_lmc

    def __call__(self):
        np.random.seed(12)
        X, Y = simulate_data(1 / 2, self.n_samples_train)
        X_out, Y_out = simulate_data(1 / 2, self.n_samples_test)
        u = simulate_uniform(1000)  # random perbutations

        # generate random classifiers
        np.random.seed(self.seed)
        ws, bs = classifiers(self.n_classifiers, symmetrized=False)

        perturbed_X = X + simulate_uniform(X.shape[0])
        # the weights of the classifers, subject to the simplex
        lbda = np.ones(self.n_classifiers) / self.n_classifiers
        list_ws = []
        list_bs = []
        list_lbda = []

        losses_in = []

        for i in range(self.nb_iter):
            # step 1. compute the (logistic) loss and gradient w.r.t. ws and bs
            adv_loss, grad_ws, grad_bs = compute_perturbed_loss(
                perturbed_X, Y, ws, bs)
            # losses_in.append(adv_loss.dot(lbda))  # save the losses

            # step 2. gradient descent for ws, bs
            list_ws.append(ws)
            list_bs.append(bs)
            lr = self.lr_position / (i + 1) ** 0.5
            ws = ws - lr * \
                (grad_ws) + np.random.normal(scale=self.noise_level *
                                             math.sqrt(lr), size=np.shape(ws))
            bs = bs - lr * \
                (grad_bs) + np.random.normal(scale=self.noise_level *
                                             math.sqrt(lr), size=np.shape(bs))
            # step 3. update the weight parameter lbda by mirror descent
            list_lbda.append(lbda)
            lbda *= np.exp(-self.lr_weight * adv_loss)
            lbda /= np.sum(lbda)
            # step 4. draw a sample from the adversary's best response distirbuiton
            perturbed_X = sample_from_logit_best_response(
                list_ws, list_bs, list_lbda, X, Y, self.epsilon, self.eta, self.n_iters_lmc, self.lr_lmc)
            # step 5. store the drawn samples (optional)

            adv_loss, _ = compute_mixed_classifer_logistic_risk_adv(
                X, Y, lbda, u, ws, bs, self.epsilon)
            losses_in.append(adv_loss)  # save the losses

        losses_in = np.array(losses_in)

        risk_nat_rand_in, _ = compute_mixed_classifer_logistic_risk_adv(
            X, Y, lbda, u, ws, bs, 0.)
        risk_adv_rand_in, _ = compute_mixed_classifer_logistic_risk_adv(
            X, Y, lbda, u, ws, bs, self.epsilon)
        risk_nat_rand_out, _ = compute_mixed_classifer_logistic_risk_adv(
            X_out, Y_out, lbda, u, ws, bs, 0.)
        risk_adv_rand_out, _ = compute_mixed_classifer_logistic_risk_adv(
            X_out, Y_out, lbda, u, ws, bs, self.epsilon)


        error_nat_rand_in, _ = compute_classification_error_adv(
            X, Y, lbda, u, ws, bs, 0.)
        error_adv_rand_in, _ = compute_classification_error_adv(
            X, Y, lbda, u, ws, bs, self.epsilon)
        error_nat_rand_out, _ = compute_classification_error_adv(
            X_out, Y_out, lbda, u, ws, bs, 0.)
        error_adv_rand_out, _ = compute_classification_error_adv(
            X_out, Y_out, lbda, u, ws, bs, self.epsilon)

        return [risk_adv_rand_in, risk_nat_rand_in, risk_adv_rand_out, risk_nat_rand_out, error_adv_rand_in, error_nat_rand_in, error_adv_rand_out, error_nat_rand_out] 


if __name__ == "__main__":
    for seed in range(0, 6):
        print('current seed: {:d}'.format(seed))
        output_file = './data/results_frat_seed_{:02d}_new.csv'.format(seed)
        results = np.zeros((51, 9))
        for i in range(51):
            eps = i * 0.1
            problem = naive_problem(100, 100, 100, 20, eps, nb_iter=1000, lr_position=1.,
                                    lr_weight=1., eta=0.01, n_iters_lmc=20, lr_lmc=0.1, seed=seed)
            curr_result = problem()
            data_to_log = np.array([[eps] + curr_result])
            results[i] = np.array([eps] + curr_result)
            i += 1
            with open(output_file, 'a') as f:
                np.savetxt(f, data_to_log, delimiter=', ')
