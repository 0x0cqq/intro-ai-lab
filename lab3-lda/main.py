# from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, wait
import math
import sys
import time
from multiprocessing import Pool

import numpy as np
from prettytable import PrettyTable
from scipy.special import gammaln  # log(gamma(x))
from scipy.special import psi  # digamma(x)
from scipy.special import logsumexp, polygamma


class LDA:
    """
    Vanilla LDA optimized with variational EM, treating topics as parameters, with scalar smoothing parameter
    """

    def __init__(self, K, V, alpha=10.0, n_jobs = 1):
        """
        Create an LDA model
        :param K: The number of topics (int)
        :param V: The size of the vocabulary
        :param alpha: Initial hyperparameter for document-topic distributions
        """
        self.K = K  # scalar number of topics
        self.V = V  # scalar size of vocabulary
        self.D = None  # scalar number of documents
        self.gammas = None  # D x K matrix of gammas
        self.log_betas = None  # V x K matrix of log(\beta)
        self.alpha = alpha  # scalar initial hyperparameter for p(\theta)
        self.bound = 0  # the variational bound
        self.token_bound = 0
        self.n_jobs = n_jobs 


    def fit(self, X, tolerance=1e-4, max_epochs=10, initial_smoothing=1.0, n_initial_docs=1,
            max_inner_iterations=20, inner_tol=1e-6, vocab=None, display_topics=False):
        """
        fit a model to data
        :param X: a list of documents (each a list of (word_index, count) tuples)
        :param tolerance: stopping criteria (relative change in bound)
        :param max_epochs: the maximum number of epochs
        :param initial_smoothing: smoothing to use when initializing topics
        :param n_initial_docs: the number of documents to use to randomly initialize each topic
        :param max_inner_iterations: maximum number of iterations for inner optimization loop (E-step)
        :param inner_tol: the tolerance for the inner optimization loop (E-step)
        :param vocab: a list of words in the vocabulary (used for displaying topics)
        :param display_topics: if True, print topics after each epoch
        :return: None
        """


        # set up initial values for monitoring convergence
        prev_bound = -np.inf
        delta = np.inf
        self.D = len(X)

        # initialize model parameters based on data
        self.init_parameters(X, initial_smoothing,
                             n_initial_docs=n_initial_docs)

        # repeat optimization until convergence
        for i in range(max_epochs):
            # update parameters (this also computes the bound)
            print("Starting epoch {}".format(i))

            self.update_parameters(
                X, max_inner_iterations=max_inner_iterations, inner_tol=inner_tol)

            # compute the relative change in the bound
            if i > 0:
                delta = (prev_bound - self.bound) / float(prev_bound)

            # print progress
            table = PrettyTable(['Epoch', 'Bound', 'LL/token', "Delta"])
            table.add_row([i, self.bound, self.token_bound / self.D, delta])
            print(table)

            # store the new value of the bound
            prev_bound = self.bound

            if vocab is not None and display_topics:
                self.print_topics(vocab)

            # check for convergence
            if (0 < delta < tolerance) or (i + 1) >= max_epochs:
                break

    def init_parameters(self, X, initital_smoothing, n_initial_docs=1):
        """
        Initialize parameters using recommended values from the original LDA paper
        :param X: the data (as above)
        :param initial_smoothing: the amount of smoothing to use in initializing the topics
        :param n_initial_docs: the number of documents to use to initialize each topic
        :return: None
        """

        phi_total = np.zeros([self.V, self.K])
        random_docs = list(np.random.choice(
            np.arange(self.D), size=n_initial_docs * self.K, replace=False))
        # initialize each topic with word counts from a subset of documents
        for k in range(self.K):
            docs = random_docs[k * n_initial_docs: (k + 1) * n_initial_docs]
            for d in docs:
                for w, c in X[d]:
                    phi_total[w, k] += c
        # smooth the counts - you should know why!
        phi_total += initital_smoothing
        # compute the corresponding topics
        self.log_betas = self.compute_log_betas_mle(phi_total)
        self.gammas = np.zeros([self.D, self.K])

    def compute_log_betas_mle(self, phi_total):
        """
        M-step for topics: compute the values of log betas to maximize the bound
        :param phi_total: np.array (V,K): Expected number of each type of token assigned to each class k
        :return: np.array (V, K): log(beta)
        """
        # sum counts over vocbaulary
        topic_word_totals = np.sum(phi_total, axis=0)
        # compute new optimal values for log betas
        log_betas = np.log(phi_total) - np.log(topic_word_totals)
        # avoid negative infinities
        log_betas[phi_total == 0] = -100
        return log_betas

    def update_parameters_for_a_doc(self, d, X, max_inner_iterations, inner_tol):
        counts_d = X[d]

        doc_phi_total = np.zeros_like(self.log_betas)

        # optimize the phi and gamma parameter for this document
        doc_bound, phi_d, gammas = self.update_parameters_for_one_item(counts_d, max_iter_d=max_inner_iterations,
                                                                       tol_d=inner_tol)
        N_d = 0
        for n, (w, c) in enumerate(counts_d):
            doc_phi_total[w, :] += c * phi_d[n, :]
            N_d += c
        doc_token_bound = doc_bound / float(N_d)
        print(".", end="")
        sys.stdout.flush()
        return gammas, doc_phi_total, doc_bound, doc_token_bound

    def update_parameters(self, X, max_inner_iterations=20, inner_tol=1e-6):
        """
        Do one epoch of updates for all parameters
        :param X: the data (D x V np.array)
        :param max_inner_iterations: the maximum number of iterations for optimizing document parameters
        :param inner_tol: the tolerance for optimizing  document parameters
        :return: None
        """
        start = time.time()
        self.bound = 0
        self.token_bound = 0
        phi_total = np.zeros_like(self.log_betas)

        # p = ProcessPoolExecutor(max_workers=N_PROCESS)
        # process = []
        # print("Total documents: {}".format(self.D))
        # for d in range(self.D):
        #     process.append(p.submit(self.update_parameters_for_a_doc, d, X, max_inner_iterations, inner_tol))
        # wait(process, return_when=ALL_COMPLETED, timeout=None)
        # for d in range(self.D):
        #     gammas, doc_phi_total, doc_bound, doc_token_bound = process[d].result()
        #     self.gammas[d, :] = gammas
        #     self.bound += doc_bound
        #     self.token_bound += doc_token_bound
        #     phi_total += doc_phi_total
        # p.shutdown()

        pool = Pool(processes=self.n_jobs)
        results = []
        for d in range(self.D):
            results.append(pool.apply_async(
                self.update_parameters_for_a_doc, (d, X, max_inner_iterations, inner_tol)))
        pool.close()
        pool.join()
        for d in range(self.D):
            gammas, doc_phi_total, doc_bound, doc_token_bound = results[d].get(
            )
            self.gammas[d, :] = gammas
            self.bound += doc_bound
            self.token_bound += doc_token_bound
            phi_total += doc_phi_total
        pool.terminate()

        # # make one update for each document
        # for d in range(self.D):
        #     if d % 10 == 0 and d > 0:
        #         print(d)
        #     counts_d = X[d]

        #     # optimize the phi and gamma parameter for this document
        #     bound, phi_d, gammas = self.update_parameters_for_one_item(counts_d, max_iter_d=max_inner_iterations,
        #                                                                tol_d=inner_tol)
        #     self.gammas[d, :] = gammas

        #     # only need to store the running sum of phi over the documents
        #     N_d = 0
        #     for n, (w, c) in enumerate(counts_d):
        #         phi_total[w, :] += c * phi_d[n, :]
        #         N_d += c

        #     # add the contribution of this document to the bound
        #     self.bound += bound
        #     self.token_bound += bound / float(N_d)

        # finally update the topic-word distributions and hyperparameters
        self.update_alpha()
        self.log_betas = self.compute_log_betas_mle(phi_total)

        print("\nTime: {}".format(time.time() - start))

    def update_parameters_for_one_item(self, count_tuples, max_iter_d=20, tol_d=1e-4):
        """
        Update gamma and compute updates for beta and the bound for one document
        :param counts: the word counts for the corresponding document (length-V np.array)
        :param max_iter_d: the maximum number of epochs for this inner optimization
        :param tol_d: the tolerance required for convergence of the inner optimization problem
        :return: (contribution to the bound, phi values for this doc, gammas for this doc)
        """

        # unzip counts into lists of word indices and counts of those words
        word_indices, counts = zip(*count_tuples)
        # convert the lists into vectors of the required shapes
        word_indices = np.reshape(
            np.array(word_indices, dtype=np.int32), (len(word_indices),))
        counts = np.reshape(
            np.array(counts, dtype=np.int32), (len(word_indices),))
        count_vector_2d = np.reshape(np.array(counts), (len(word_indices), 1))

        # count the total number of words
        N_d = int(count_vector_2d.sum())
        # and the number of distinct word types
        n_word_types = len(word_indices)

        # initialize gamma values to alpha + 1/K
        gammas = self.alpha + N_d * np.ones(self.K) / float(self.K)
        # initialize phis to 1/K; only need to consider each word type
        phi_d = np.ones([n_word_types, self.K]) / float(self.K)

        # do the optimization step
        bound, new_gammas, new_phi_d = update_params_for_one_item(
            self.K, n_word_types, word_indices, counts, gammas, phi_d, self.log_betas, self.alpha, max_iter_d, tol_d)

        return bound, new_phi_d, new_gammas

    def update_alpha(self, newton_thresh=1e-5, max_iter=1000):
        """
        Update hyperparameters of p(\theta) using Netwon's method [ported from lda-c]
        :param newton_thresh: tolerance for Newton optimization
        :param max_iter: maximum number of iterations
        :return: None
        """

        init_alpha = self.alpha
        log_alpha = np.log(init_alpha)

        psi_gammas = psi(self.gammas)
        psi_sum_gammas = psi(np.sum(self.gammas, axis=1))
        # print(psi_gammas)
        # print(psi_sum_gammas)
        # print(np.reshape(psi_sum_gammas, (self.D, 1)))
        E_ln_thetas = psi_gammas - np.reshape(psi_sum_gammas, (self.D, 1))
        # print(E_ln_thetas)
        # called ss (sufficient statistics) in lda-c
        sum_E_ln_theta = np.sum(E_ln_thetas)
        # print(sum_E_ln_theta)
        print("maximization L(alpha): ")
        # repeat until convergence
        table = PrettyTable(["alpha", "L(alpha)", "dL(alpha)"])
        data = []
        for i in range(max_iter):
            alpha = np.exp(log_alpha)
            if np.isnan(alpha):
                init_alpha *= 10
                print("warning : alpha is nan; new init = %0.5f" % init_alpha)
                alpha = init_alpha
                log_alpha = np.log(alpha)

            L_alpha = self.compute_L_alpha(alpha, sum_E_ln_theta)
            dL_alpha = self.compute_dL_alpha(alpha, sum_E_ln_theta)
            d2L_alpha = self.compute_d2L_alpha(alpha)
            log_alpha = log_alpha - dL_alpha / (d2L_alpha * alpha + dL_alpha)

            table.add_row([alpha, L_alpha, dL_alpha])
            # print("%5.5f\t%5.5f\t%5.5f" % (np.exp(log_alpha), L_alpha, dL_alpha))
            if np.abs(dL_alpha) <= newton_thresh:
                break
        print(table)
        self.alpha = np.exp(log_alpha)

    def compute_L_alpha(self, alpha, sum_E_ln_theta):
        return self.D * (gammaln(self.K * alpha) - self.K * gammaln(alpha)) + (alpha - 1) * sum_E_ln_theta

    def compute_dL_alpha(self, alpha, sum_E_ln_theta):
        return self.D * (self.K * psi(self.K * alpha) - self.K * psi(alpha)) + sum_E_ln_theta

    def compute_d2L_alpha(self, alpha):
        return self.D * (self.K ** 2 * polygamma(1, self.K * alpha) - self.K * polygamma(1, alpha))

    def print_topics(self, vocab, n_words=8):
        """
        Display the top words in each topic
        :param vocab: a list of words in the vocabulary
        """
        table = PrettyTable(["Topic"] + ["Word %d" % (i + 1)
                            for i in range(n_words)])
        for k in range(self.K):
            order = list(np.argsort(self.log_betas[:, k]).tolist())
            order.reverse()
            table.add_row([k] + [vocab[i] for i in order[:n_words]])
            # print("%d %s" % (k, ' '.join([vocab[i] for i in order[:n_words]])))
        print(table)


def update_params_for_one_item(K, n_word_types, word_indices, counts, gammas, phi_d, log_betas, alpha, max_iter_d,
                               tol_d):
    """
    Optimize the per-document variational parameters for one document, namely gamma and phi.
    :param K: the number of topics
    :param n_word_types: the number of word types in this document (length of word_indices)
    :param word_indices: a typed memory view of the indices of the words in the document
    :param counts: the corresponding counts of each word in the document
    :param gammas: the variational parameters of this document to be updated (length-K)
    :param phi_d: n_word_types x K memory view of expected distribution of topics for each word
    :param log_betas: V x K memoryview of the current value of the log of the topic distributions
    :param alpha: current value of the hyperparmeter alpha
    :param max_iter_d: the maximum number of iterations for this inner optimization loop
    :param tol_d: the tolerance required for convergence of this inner optimization loop
    """
    i = 0
    prev_bound = -1000000.0
    bound = 0.0
    delta = tol_d

    psi_gammas = psi(gammas)
    # print("gammas: ", gammas)

    # repeat until convergence
    while i < max_iter_d and delta >= tol_d:
        # process all the word index: count pairs in this documents
        for n in range(n_word_types):
            w = word_indices[n]
            c = counts[n]
            new_phi_dn = np.zeros(K)

            ##########################################################################
            # TODO: update variational parameters inplace: gamma and phi             #
            #                                                                        #
            #                                                                        #
            #                                                                        #
            ##########################################################################

            # Update phi
            # Calculate log phi_dn
            # print("n:", n)
            for j in range(K):
                new_phi_dn[j] = log_betas[w, j] + psi_gammas[j]
            # Normalize them to sum 1
            new_phi_dn -= logsumexp(new_phi_dn)

            # update to phi_d
            phi_d[n] = np.exp(new_phi_dn)

        # Update gamma
        # print("alpha:", alpha)
        for i in range(K):
            gammas[i] = alpha

        for n in range(n_word_types):
            c = counts[n]
            for i in range(K):
                gammas[i] += c * phi_d[n, i]

        # update psi_gamma
        psi_gammas = psi(gammas)

        # compute the part of the variational bound corresponding to this document
        bound = compute_bound_for_one_item(
            K, n_word_types, word_indices, counts, alpha, gammas, psi_gammas, phi_d, log_betas)

        # compute the relative change in the bound
        delta = (prev_bound - bound) / prev_bound

        # save the new value of the bound
        prev_bound = bound
        i += 1

    return bound, gammas, phi_d


def compute_bound_for_one_item(K, n_word_types, word_indices, count_vector, alpha, gammas, psi_gammas, phi_d,
                               log_betas):
    """
    Compute the parts of the variational bound corresponding to the particular document
    :param K: the number of topics
    :param n_word_types: the number of word types in this document (length of count_vector)
    :param word_indices: a vector of vocabulary indices of the word types in this document
    :param count_vector: the corresponding vector of counts of each word type
    :param alpha: the current value of the hyperparameter alpha
    :param gammas: the current value of gammas for this document
    :param psi_gammas: pre-computed values of psi(gammas)
    :param phi_d: the expected distribution of topics for each word type in this document
    :param log_betas: the current value of the log of the topic distributions
    """

    bound = 0.0

    ##########################################################################
    # TODO: calculate ELBO, return it as bound                               #
    #                                                                        #
    #                                                                        #
    #                                                                        #
    ##########################################################################

    # psisumgammas = psi(np.sum(gammas))

    bound += gammaln(K * alpha) - np.sum(K * gammaln(alpha)) + \
        np.sum((alpha - 1) * (psi_gammas - psi(np.sum(gammas))))

    for n in range(n_word_types):
        for i in range(K):
            bound += phi_d[n, i] * (psi_gammas[i] -
                                    psi(np.sum(gammas))) * count_vector[n]
            bound += log_betas[word_indices[n], i] * \
                phi_d[n, i] * count_vector[n]

    bound += -gammaln(np.sum(gammas)) + np.sum(gammaln(gammas))

    for i in range(K):
        bound -= (gammas[i] - 1) * (psi_gammas[i] - psi(np.sum(gammas)))

    for n in range(n_word_types):
        for i in range(K):
            bound -= phi_d[n, i] * \
                math.log(phi_d[n, i] + 1e-10) * count_vector[n]

    return bound


if __name__ == "__main__":
    from utils import preprocess
    K = 5
    N_PROCESS = 6
    print(f"K:{K} N_PROCESS:{N_PROCESS}")
    docs, _, vocab = preprocess("./dataset/dataset_cn_full.txt")
    lda = LDA(K=K, V=len(vocab), n_jobs=N_PROCESS)
    time_start = time.time()
    lda.fit(docs, vocab=vocab, display_topics=True)
    print(f"Total time taken: {time.time() - time_start}")
