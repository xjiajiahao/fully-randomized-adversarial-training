import numpy as np
from comm import *


class naive_problem:
    def __init__(self, n_samples_train, n_samples_test, n_classifiers_candidates, n_classifiers, epsilon,
                 nb_iter=10000, lr_position=2., lr_weight=1., n_iters_attack=20, lr_attack=0.1, seed=12):
        self.seed = seed
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.n_classifiers_candidates = n_classifiers_candidates
        self.n_classifiers = n_classifiers
        self.epsilon = epsilon
        self.nb_iter = nb_iter
        self.lr_position = lr_position
        self.lr_weight = lr_weight
        self.n_iters_attack = n_iters_attack
        self.lr_attack = lr_attack

    def __call__(self):
        np.random.seed(12)
        X, Y = simulate_data(1 / 2, self.n_samples_train)
        X_out, Y_out = simulate_data(1 / 2, self.n_samples_test)
        u = simulate_uniform(1000)  # random perbutations

        # generate random classifiers
        np.random.seed(self.seed)
        ws, bs = classifiers(1, symmetrized=False)
        lbda = [1.0]

        perturbed_X = X + simulate_uniform(X.shape[0])
        # the weights of the classifers, subject to the simplex

        losses_in = []

        for i in range(self.nb_iter):
            # step 1. PGD attack for multiple iterations
            perturbed_X = np.copy(X)
            if self.epsilon > 0:
                for _ in range(self.n_iters_attack):
                    adv_loss, grad_X = compute_grad_X_perturbed_loss(perturbed_X,
                                                                     Y, ws, bs)
                    perturbed_X += self.lr_attack * grad_X
                    scale = np.minimum(self.epsilon / np.sqrt(np.sum((perturbed_X - X)**2, axis=1)), 1.0)
                    perturbed_X = X + ((perturbed_X - X).T * scale).T

            # step 2. gradient descent for ws, bs
            # lr = self.lr_position
            # ws = ws - lr * (grad_ws)
            # bs = bs - lr * (grad_bs)
            adv_loss, grad_ws, grad_bs = compute_perturbed_loss(
                perturbed_X, Y, ws, bs)
            lr = self.lr_position / (i + 1) ** 0.5
            ws = ws - lr * grad_ws
            bs = bs - lr * grad_bs

            adv_loss, _ = compute_mixed_classifer_logistic_risk_adv(
                X, Y, lbda, u, ws, bs, self.epsilon)
            losses_in.append(adv_loss )  # save the losses

        losses_in = np.array(losses_in)

        # construct the second component
        # 1. find the best adv for the first component
        perturbed_X = np.copy(X)
        for tmpi, (x, y) in enumerate(zip(X, Y)):
            x_candidates = x + self.epsilon * u  # x: 1-by-2, u: num_random_perturbations-by-2
            z = ((np.matmul(X, ws.T) + bs).T * y).T  # num_random_perturbations-by-n_classifiers
            sign_z = np.sign(z)
            loss = np.log(1 + np.exp(-np.abs(z))) + (sign_z - 1) / 2 * z
            mixed_loss = loss.dot(lbda)
            adv_index = np.argmax(mixed_loss)  # the worst average loss
            perturbed_X[tmpi] = x_candidates[adv_index]
        # 2. train a classifier on perturbed_X
        ws_second = ws.copy()
        bs_second = bs.copy()
        for i in range(self.nb_iter):
            # lr = self.lr_position
            # ws_second = ws_second - lr * (grad_ws_second)
            # bs_second = bs_second - lr * (grad_bs_second)
            adv_loss_second, grad_ws_second, grad_bs_second = compute_perturbed_loss(
                perturbed_X, Y, ws_second, bs_second)
            lr = self.lr_position / (i + 1) ** 0.5
            ws_second = ws_second - lr * grad_ws_second
            bs_second = bs_second - lr * grad_bs_second
        ws = np.concatenate((ws, ws_second), axis=0)
        bs = np.concatenate((bs, bs_second), axis=0)
        # 3. find the best lbda
        best_lbda = None
        opt_adv_loss = math.inf
        for i in range(0, 101):
            tmp_lbda = np.array([i*0.01, (100 - i)*0.01])
            tmp_adv_risk, _ = compute_mixed_classifer_logistic_risk_adv(
                              X, Y, tmp_lbda, u, ws, bs, self.epsilon)
            if tmp_adv_risk < opt_adv_loss:
                opt_adv_loss = tmp_adv_risk
                best_lbda = tmp_lbda
        lbda = best_lbda

        # np.savetxt(f"training_adv_losses_{self.epsilon}.txt", losses_in)

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
        output_file = './data/results_bat_seed_{:02d}.csv'.format(seed)
        results = np.zeros((51, 9))
        for i in range(51):
        # for i in range(30, 31):
            eps = i * 0.1
            if seed == 6 or seed >= 8 and seed <= 9:
                lr_position = 5.0
                lr_weight = 5.0
            else:
                lr_position = 1.0
                lr_weight = 1.0
            problem = naive_problem(100, 100, 100, 20, eps, nb_iter=1000, lr_position=lr_position,
                                    lr_weight=lr_weight, n_iters_attack=20, lr_attack=1., seed=seed)
            curr_result = problem()
            data_to_log = np.array([[eps] + curr_result])
            results[i] = np.array([eps] + curr_result)
            i += 1
            with open(output_file, 'a') as f:
                np.savetxt(f, data_to_log, delimiter=', ')

        # plt.figure()
        # plt.plot(results[:, 0], results[:, 1])
        # plt.show()
