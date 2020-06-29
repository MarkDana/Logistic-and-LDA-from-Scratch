# Created by Haoyue Dai@06/27/2020

import numpy as np
import utils
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse

class LogisticRegression(object):
    # this can suit for both binary classification and multi classification
    # binary: self.num_classes=1, outputs, predictions, targets in shape (n,1)
    # numti-k-classes: self.num_classes=k, outputs, predictions, targets in shape (n,k)
    def __init__(self, lr=0.01, epochs=1000, add_bias=True, logdir=None, lossname='MLE', regular_lambda=0., use_kernel=None,
                 load_pre_kernel=True, load_pre_kernel_from='./precomputed_kernels/', kernel_lasso=False, rbf_sigma=1, poly_d=2, sigmoid_alpha=1e-4, sigmoid_c=1e-2):
        self.num_classes = 1
        self.lr = lr
        self.epochs = epochs
        self.lossname = lossname
        self.regular_lambda = regular_lambda
        self.use_kernel = use_kernel
        self.load_pre_kernel = load_pre_kernel
        self.load_pre_kernel_from = load_pre_kernel_from
        os.makedirs(load_pre_kernel_from, exist_ok=True)
        self.kernel_lasso = kernel_lasso
        self.rbf_sigma = rbf_sigma
        self.poly_d = poly_d
        self.sigmoid_alpha = sigmoid_alpha
        self.sigmoid_c = sigmoid_c
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.add_bias = add_bias
        self.logdir = logdir
        if logdir: os.makedirs(logdir, exist_ok=True)

    def load(self, beta_npy_pth):
        self.beta = np.load(beta_npy_pth)

    def softmax(self, outputs):
        exp_outputs = np.exp(outputs)
        return exp_outputs / np.sum(exp_outputs, axis=1)[:, None]

    def add_bias_to_vector(self, vectors):
        bias = np.ones((vectors.shape[0], 1))
        return np.concatenate((bias, vectors), axis=1)

    def precompute_kernel(self, vectors_i, vectors_j):
        numi, numj = len(vectors_i), len(vectors_j)
        nameflag = 'train' if numi == numj else 'test' #assume training samples num != test samples num
        if self.load_pre_kernel_from:
            try:
                return np.load(os.path.join(self.load_pre_kernel_from, f'{self.use_kernel}_{nameflag}.npy'))
            except:
                print('Precomputed kernel matrix not found, now starts computing')

        def _rbf(vectors_i, vectors_j):
            # diff_norm[i,j]=norm_2(vectors_i[i]-vectors_j[j])^2
            diff_norm = np.linalg.norm(vectors_i[:, None] - \
                                       vectors_j[None, :], axis=2) ** 2
            return np.exp(-0.5 * diff_norm / self.rbf_sigma ** 2)

        def _poly(vectors_i, vectors_j):
            # inner_product in shape [I,J], with matmul[i,j]=vectors_i[i].T dot vectors_j[j]
            inner_product = np.matmul(vectors_i[:, None, None],
                    vectors_j[None, :, :, None]).reshape((numi, numj))
            return inner_product ** self.poly_d

        def _cos(vectors_i, vectors_j):
            inner_product = np.matmul(vectors_i[:, None, None],
                        vectors_j[None, :, :, None]).reshape((numi, numj))
            norm_product = \
                (np.linalg.norm(vectors_i, axis=1, keepdims=True)[:, None]
                * np.linalg.norm(vectors_j, axis=1, keepdims=True)[None, :])\
                .reshape((numi, numj))
            return inner_product / norm_product

        def _sigmoid(vectors_i, vectors_j):
            tobe_tanh = self.sigmoid_alpha * np.matmul(
                vectors_i[:, None, None], vectors_j[None, :, :, None]) + self.sigmoid_c
            exp_tobe_tanh = np.exp(-2 * tobe_tanh.reshape((numi, numj)))
            return (1 - exp_tobe_tanh) / (1 + exp_tobe_tanh)

        returnDict = {'RBF': _rbf, 'poly': _poly, 'cos': _cos, 'sigmoid': _sigmoid}
        kernel_mat = returnDict[self.use_kernel](vectors_i, vectors_j)
        kernel_mat = (kernel_mat - kernel_mat.mean()) / kernel_mat.std() #normalize
        np.save(os.path.join(self.load_pre_kernel_from, f'{self.use_kernel}_{nameflag}.npy'), kernel_mat)
        return returnDict[self.use_kernel](vectors_i, vectors_j)


    def loss(self, outputs, predictions, targets):
        def _neg_log_likelihood(outputs, predictions, targets):
            # default MLE: -1 * log likelihood of labels of all samples being correctly estimated
            if self.num_classes == 1:
                # \mean(y_i * X_i^T \beta - \log(1 + \exp(X_i^T \beta)))
                return -1. * np.mean(targets * outputs - np.log(1 + np.exp(outputs)))
            else:
                # equivalent to negative cross entropy loss \mean_i {y_i \log(p_i)}
                gt_classes = np.where(targets == 1) # ground-truth label
                return -1. * np.mean(np.log(predictions[gt_classes]))

        def _ridge_loss(outputs, predictions, targets):
            # penalization term is \lambda / 2 * ||beta||_2^2
            return _neg_log_likelihood(outputs, predictions, targets) + 0.5 * self.regular_lambda * np.linalg.norm(self.beta) ** 2

        def _lasso_loss(outputs, predictions, targets):
            # penalization term is \lambda * ||beta||_1
            return _neg_log_likelihood(outputs, predictions, targets) + self.regular_lambda * np.linalg.norm(self.beta, ord=1)

        returnDict = {'MLE': _neg_log_likelihood, 'Ridge': _ridge_loss, 'LASSO': _lasso_loss}
        return returnDict[self.lossname](outputs, predictions, targets)

    def gradient(self, vectors, targets, predictions):
        # same for binary or multiple classification
        # learn from mistakes: mean_i((p_i - y_i) X_i), for loss decay
        def _grad_neg_log_likelihood(vectors, targets, predictions):
            return np.mean(np.matmul(vectors[:, :, None],
                        (predictions - targets)[:, None, :]), axis=0)

        def _grad_ridge_loss(vectors, targets, predictions):
            one_if_positive_else_minus_one = (self.beta > 0) * 2 - 1
            return _grad_neg_log_likelihood(vectors, targets, predictions) + self.regular_lambda * one_if_positive_else_minus_one

        def _grad_lasso_loss(vectors, targets, predictions):
            return _grad_neg_log_likelihood(vectors, targets, predictions)\
                   + self.regular_lambda * self.beta

        returnDict = {'MLE': _grad_neg_log_likelihood, 'Ridge': _grad_ridge_loss, 'LASSO': _grad_lasso_loss}
        return returnDict[self.lossname](vectors, targets, predictions)


    def accuracy(self, predictions, targets):
        if self.num_classes == 1:
            return np.mean(np.round(predictions) == targets)
        else:
            return np.mean(np.argmax(predictions, axis=1) == \
                           np.argmax(targets, axis=1))


    def forward(self, vectors, targets, from_raw=False):
        if from_raw: vectors = self.add_bias_to_vector(vectors)
        outputs = np.dot(vectors, self.beta)
        predictions = self.sigmoid(outputs) if self.num_classes == 1 \
            else self.softmax(outputs)
        loss = self.loss(outputs, predictions, targets)
        accuracy = self.accuracy(predictions, targets)
        return outputs, predictions, loss, accuracy

    def fit(self, train_vectors, train_targets, test_vectors, test_targets):
        if self.add_bias:
            train_vectors = self.add_bias_to_vector(train_vectors)
            test_vectors = self.add_bias_to_vector(test_vectors)
        if self.use_kernel:
            train_kernel_mat = self.precompute_kernel(train_vectors, train_vectors)
            test_kernel_mat = self.precompute_kernel(test_vectors, train_vectors)
            train_vectors, test_vectors = train_kernel_mat, test_kernel_mat

        self.num_classes = train_targets.shape[1]
        self.beta = np.zeros((train_vectors.shape[1], self.num_classes))

        best_test_acc = 0
        for i in range(self.epochs):
            train_outputs, train_predictions, train_loss, train_accuracy = self.forward(train_vectors, train_targets)
            gradient = self.gradient(train_vectors, train_targets, train_predictions)

            if self.use_kernel:
                if self.kernel_lasso:
                    gradient += self.regular_lambda * self.beta
                else:
                    gradient += self.regular_lambda * np.matmul(train_vectors, self.beta)

            # print('grad', self.beta.mean(), gradient.mean())
            self.beta -= self.lr * gradient

            test_outputs, test_predictions, test_loss, test_accuracy = self.forward(test_vectors, test_targets)
            print(f'Epoch[{i}/{self.epochs}]\tTest-Loss[{test_loss}]\tTest-Accuracy[{test_accuracy}]')

            if self.logdir:
                with open(os.path.join(self.logdir, 'train_loss_acc.txt'), 'a') as fout:
                    fout.write(f'{i}\t{train_loss}\t{train_accuracy}\n')
                with open(os.path.join(self.logdir, 'test_loss_acc.txt'), 'a') as fout:
                    fout.write(f'{i}\t{test_loss}\t{test_accuracy}\n')
                with open(os.path.join(self.logdir, 'beta_mean_var.txt'), 'a') as fout:
                    fout.write(f'{i}\t{self.beta.mean()}\t{self.beta.var()}\n')
                if test_accuracy > best_test_acc:
                    best_test_acc = test_accuracy
                    np.save(os.path.join(self.logdir, 'bestbeta.npy'), self.beta)
    # |========================================================================|
    # |============== Codes below are for visualizaiton and test ==============|
    # |========================================================================|
    def drawTwoLoss(self, result_dir):
        print('=' * 30 + f' now drawing loss curves for {result_dir} ' + '=' * 30)
        utils.drawCurveDonkey(os.path.join(result_dir, 'train_loss_acc.txt'), os.path.join(result_dir, 'train.png'), title=f"train: {os.path.basename(os.path.normpath(result_dir))}")
        utils.drawCurveDonkey(os.path.join(result_dir, 'test_loss_acc.txt'), os.path.join(result_dir, 'test.png'), title=f"test: {os.path.basename(os.path.normpath(result_dir))}")

    def vis_weight(self, result_dir, digits_name=None, tight=False):
        if not digits_name: digits_name = [0, 1]
        weights = np.load(os.path.join(result_dir, 'bestbeta.npy'))
        if not tight:
            if self.num_classes == 1:
                plt.imshow(weights[-784:].reshape([28, 28]), cmap='gray')
                plt.title(f'Weight of LDA, binary on {digits_name}')
            else:
                for i in range(self.num_classes):
                    plt.subplot(2, self.num_classes / 2, i + 1)
                    weight = weights[-784:, i].reshape([28, 28])
                    plt.title(i)
                    plt.imshow(weight, cmap='gray') # 'RdBu' is also ok; 'gray' makes the numbers stand out more
                    frame1 = plt.gca()
                    frame1.axes.get_xaxis().set_visible(False)
                    frame1.axes.get_yaxis().set_visible(False)
        else:
            fig = plt.figure(figsize=(self.num_classes, 1))
            gs = gridspec.GridSpec(1, self.num_classes, width_ratios=[1, ] * self.num_classes,
                                   wspace=0.0, hspace=0.0, top=1., bottom=0., left=0.,
                                   right=1.)

            for i in range(self.num_classes):
                im = weights[-784:, i].reshape([28, 28])
                ax = plt.subplot(gs[0, i])
                ax.imshow(im, cmap='gray')
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        plt.savefig(os.path.join(result_dir, f'visbeta{"_tight" if tight else ""}.png'), dpi=150)
        plt.clf()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Logistic Regression')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1000, type=int, help='epochs')
    parser.add_argument('--loss', default='MLE', type=str, help='MLE/Ridge/LASSO')
    parser.add_argument('--regualr_lambda', 'lam', default=0., type=float, help='regularization term')
    parser.add_argument('--kernel', default=None, type=str, help='RBF, poly, cos, sigmoid')

    parser.add_argument('--sample10class', '-sub', action='store_true', help='whether to sample a subset')
    parser.add_argument('--traintest', nargs='+', type=int, help='#load training&testing samples')

    parser.add_argument('--sample1vsrest', '-1vsr', action='store_true', help='whether to sample one class and rest')
    parser.add_argument('--loadone', default=1, type=int, help='#load 1vsrest')

    parser.add_argument('--sample1vs1', '-1vs1', action='store_true', help='whether to sample two classes')
    parser.add_argument('--loadtwo', nargs='+', type=int, help='#load 1vs1')

    args = parser.parse_args()

    sample_name_flag = 'fulldata'
    if args.sample10class:
        train_images, train_labels, test_images, test_labels = utils.load_MNIST_np(
            sample_train_test=tuple(args.traintest))
        sample_name_flag = f'subset_{"_".join(tuple(args.traintest))}'
    elif args.sample1vsrest:
        train_images, train_labels, test_images, test_labels = utils.load_MNIST_np(
            load_binary=args.loadone)
        sample_name_flag = f'{args.loadone}_rest'
    elif args.sample1vs1:
        train_images, train_labels, test_images, test_labels = utils.load_MNIST_np(
            load_two_classes=tuple(args.loadtwo))
        sample_name_flag = f'{"_".join(tuple(args.loadtwo))}'
    else:
        train_images, train_labels, test_images, test_labels = utils.load_MNIST_np()

    logdir = f'./LRresults/loss_{args.loss}_kernel_{args.kernel}_sample_{sample_name_flag}'
    logistic = LogisticRegression(epochs=args.epochs, lr=args.lr, logdir=logdir,
                      lossname=args.loss, regular_lambda=args.regualr_lambda, use_kernel=args.kernel)
    logistic.fit(train_images, train_labels, test_images, test_labels)
    logistic.drawTwoLoss(logdir)
    logistic.vis_weight(logdir, tight=True)


    # # example to run cross entropy 10-classification, samples only 5000, 1000 images
    # train_images, train_labels, test_images, test_labels = utils.load_MNIST_np(sample_train_test=(5000, 1000))
    # logdir = './simple_mle_10'
    #
    #
    # # example to run binary classification, samples 0-1 (or use load_binary=1 for 1-rest)
    # train_images, train_labels, test_images, test_labels = utils.load_MNIST_np(load_two_classes=(0, 1))
    # logdir = './simple_mle_binary'
    # logistic = LogisticRegression(epochs=1000, lr=0.01, logdir=logdir)
    # logistic.fit(train_images, train_labels, test_images, test_labels)
    # logistic.drawTwoLoss(logdir)
    # logistic.vis_weight(logdir, tight=True)
    #
    # # example to use Ridge regularization
    # # (default MLE, options are Ridge and LASSO)
    # train_images, train_labels, test_images, test_labels = utils.load_MNIST_np()
    # logdir = './ridge_on_10'
    # logistic = LogisticRegression(epochs=1000, lr=0.01, logdir=logdir, lossname='Ridge', regular_lambda=0.01)
    # logistic.fit(train_images, train_labels, test_images, test_labels)
    # logistic.drawTwoLoss(logdir)
    # logistic.vis_weight(logdir, tight=True)
    #
    # # example to use kernel based ridge regression
    # # (options are RBF, poly, cos, sigmoid)
    # # you may add lossname='LASSO' to enable regularization, or adjust kernel hyperparameters. see init above
    # # on your first run, kernels will be computed, which may take a while.
    # # afterwards, precomputed kernels are saved in ./precomputed_kernels and will be auto loaded
    # train_images, train_labels, test_images, test_labels = utils.load_MNIST_np(sample_train_test=(1000, 500))
    # logdir = './ridge_on_10'
    # logistic = LogisticRegression(epochs=1000, lr=0.01, logdir=logdir, use_kernel='RBF')
    # logistic.fit(train_images, train_labels, test_images, test_labels)
    # logistic.drawTwoLoss(logdir)
    # logistic.vis_weight(logdir, tight=True)
