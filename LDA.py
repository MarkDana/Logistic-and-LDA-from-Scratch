# Created by Haoyue Dai@06/27/2020

import numpy as np
import argparse
import utils
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import matplotlib.cm as cm
plt.rcParams.update({'figure.max_open_warning': 0})
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

IMGSHAPE = 28
class LDA(object):
    # this can suit for both binary classification and multi classification
    # binary: self.num_classes=2, self.project_dims = 1, outputs in shape (n,1), use threshold
    # numti-k-classes: self.num_classes=k, self.project_dims = k, outputs in shape (n,k), use gaussian
    def __init__(self,):
        self.num_classes = 2 # or 10
        self.project_dims = 1 # or 10
        self.digits_name = list(range(10))

    def from_one_col_to_binary(self, targets):
        # transform the load_binary [1 0 1] -> [[0 1 0] [1 0 1]]
        return np.concatenate([1 - targets, targets], axis=1)

    def estimate(self, train_vectors, train_targets):
        self.num_classes = train_targets.shape[1]
        means = np.zeros((self.num_classes, train_vectors.shape[1]))
        indices_of_classes_train = []
        means_all = np.mean(train_vectors, axis=0)
        for i in range(self.num_classes):
            indices = np.where(train_targets[:, i] == 1)
            indices_of_classes_train.append(indices)
            means[i] = np.mean(train_vectors[indices], axis=0)
        self.means = means
        self.indices_of_classes_train = indices_of_classes_train

        # between class covariance S_B = sum (\mu_c-\mu) (\mu_c-\mu).T, add by all samples
        normed_means = means - means_all
        S_B = np.matmul(normed_means[:, :, None], normed_means[:, None, :]) # [10, 784, 784], with S_B[i]=np.outer(normed_means[i], normed_means[i])
        S_B *= np.array([len(x[0]) for x in
                self.indices_of_classes_train])[:, None, None] # weighted by samples num
        S_B = np.sum(S_B, axis=0) # sum after weighted

        # within class covariance S_W = \sum (\x_c-\mu_c) (\x_c-\mu_c).T
        S_W = np.zeros_like(S_B)
        for i in range(self.num_classes):
            tmp = train_vectors[indices_of_classes_train[i]] - means[i]
            S_W += np.matmul(tmp.T, tmp)

        # objective: find eigenvalue, eigenvector pairs for inv(S_W).S_B
        mat = np.matmul(np.linalg.pinv(S_W), S_B)
        # sort the eigvals in decreasing order, take the first project_dims eigvectors
        eigvals, eigvecs = np.linalg.eig(mat)
        eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]
        eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)
        self.weight = np.real(np.array([eiglist[i][1]
                    for i in range(self.project_dims)]).T)
        # complex to real (complex from eigvecs)
        # in shape (784, project_dims)

    def cal_acc_by_threshold(self, vectors, targets, istestset=False): #works for binary classification (project to 1-dim)
        mean_proj = np.matmul(self.means, self.weight) #eg. [[0.9] [0.3]] in shape (2,1)
        threshold_proj = np.mean(mean_proj)
        self.threshold_proj = threshold_proj
        which_is_larger = np.argmax(mean_proj) # int 0 or 1, for above example: 0
        data_proj = np.matmul(vectors, self.weight).flatten() # e.g. [0.65, 0.33, 0.1, ...] in shape (n,)
        data_larger = data_proj > threshold_proj #eg. [True, False, False] in shape (n,)
        gt_labels = targets[:, which_is_larger].astype(bool)

        return np.mean(gt_labels == data_larger)

    def probabilities(self, points, mean, cov):
        '''
        return the probability densities (pdf) for given points (shape (n, 10)) on N(mean, cov)
        :param points: shape (n, 10) # here 10 is the projected dimension
        :param mean: shape (10, )
        :param cov: shape (10, 10)
        :return: shape (n, )
        '''
        cons = 1. / ((2 * np.pi) ** (points.shape[1] / 2.) *
                     np.linalg.det(cov) ** (-0.5))
        submean = points - mean
        return cons * np.exp(-np.matmul(np.matmul(submean, np.linalg.inv(cov))
                    [:, None, :], submean[:, :, None]) / 2.).flatten()

    def cal_acc_by_gaussian(self, vectors, targets, istestset=False): #works for multiclass classification (project to k-dim)

        indices_of_classes = [np.where(targets[:, i] == 1) for i in range(self.num_classes)] if istestset \
            else self.indices_of_classes_train

        data_proj = np.matmul(vectors, self.weight) # e.g. in shape (n, 10)

        # priors in shape (10, )
        # gaussian_means in shape (10, 10)
        # gaussian_covs in shape (10, 10, 10)
        priors = np.array([vectors[indices_of_classes[i]].shape[0] / vectors.shape[0] for i in range(self.num_classes)])
        gaussian_means = np.array([np.mean(data_proj[indices_of_classes[i]], axis=0) for i in range(self.num_classes)])
        gaussian_covs = np.array([np.cov(data_proj[indices_of_classes[i]], rowvar=False) for i in range(self.num_classes)])
        self.gaussian_means = gaussian_means
        self.gaussian_covs = gaussian_covs
        # for each class i, calculate all samples' probability over i's gaussian model distribution, in shape (n, 10)
        likelihoods = np.array([priors[i] * self.probabilities(data_proj, gaussian_means[i], gaussian_covs[i])
                                for i in range(self.num_classes)]).T

        predictions = np.argmax(likelihoods, axis=1)
        gt_labels = np.argmax(targets, axis=1) # where dim=1

        return np.mean(predictions == gt_labels)

    def fit(self, train_vectors, train_targets, test_vectors, test_targets, two_digits_name=None):
        if two_digits_name: self.digits_name = two_digits_name
        self.project_dims = train_targets.shape[1] #project to 1-dim if binary; else eg. 10
        if train_targets.shape[1] == 1: train_targets = self.from_one_col_to_binary(train_targets)
        if test_targets.shape[1] == 1: test_targets = self.from_one_col_to_binary(test_targets)
        self.estimate(train_vectors, train_targets)
        print(f'Weight already estimated, in shape {self.weight.shape}')

        acc_func = self.cal_acc_by_threshold if self.project_dims == 1 else self.cal_acc_by_gaussian
        train_acc = acc_func(train_vectors, train_targets)
        test_acc = acc_func(test_vectors, test_targets, istestset=True)
        print(f'Train accuracy = {train_acc}. Test accuracy = {test_acc}')
        return train_acc, test_acc

    #|========================================================================|
    #|============== Codes below are for visualizaiton and test ==============|
    #|========================================================================|

    def plot_bivariate_gaussians(self, savepth, picktwo=None):
        # savepth is without suffix
        if not picktwo: picktwo = [3,6]
        cmap = cm.rainbow(np.linspace(0, 1, len(self.digits_name)))
        fig1 = plt.figure(1)
        ax3D = fig1.gca(projection='3d')

        fig2 = plt.figure(2)
        ax2D = fig2.subplots()

        fake2Dlines = []
        labels = [f'{x}' for x in self.digits_name]
        for i in range(len(self.digits_name)):
            print(f'now plotting pdfs for digit {i}')
            mean = self.gaussian_means[i][picktwo]
            cov = self.gaussian_covs[i][picktwo, :][:, picktwo] #only pick two dimensions
            data = np.random.multivariate_normal(mean, cov, size=100)
            X, Y = np.meshgrid(data[:, 0], data[:, 1])
            points = np.stack([np.ravel(X), np.ravel(Y)]).T
            pdfs = self.probabilities(points, mean, cov)
            Z = pdfs.reshape(X.shape)
            colorRGBA = cmap[i]
            surf = ax3D.plot_surface(X, Y, Z, rstride=1, alpha=0.6, cstride=1,
                                     color=colorRGBA, linewidth=0,
                                     antialiased=False,)
            fake2Dlines.append(mpl.lines.Line2D([0],[0], linestyle="none", c=colorRGBA, marker = 'o'))

            cmap_transparency_arr = np.repeat(colorRGBA[None, :], 30, axis=0) # value numbers
            for colordim in range(4):
                cmap_transparency_arr[:, colordim] = np.linspace(cmap_transparency_arr[0, colordim], 0, 30)
            cmap_transparency = ListedColormap(cmap_transparency_arr)
            ax2D.contourf(X, Y, Z, cmap=cmap_transparency,)
            # print(cmap_transparency.colors)
            # print(cm.get_cmap('Blues', 30)(range(30)))
        ax3D.legend(fake2Dlines, labels)

        # ax2D.colorbar()
        fig2.title(f'Distribution of classes\' Gaussian model, on {picktwo} dimensions')
        fig2.savefig(savepth + '2D_gaussian_contour.png', dpi=150)  # dpi=150

        fig1.suptitle(f'Distribution of classes\' Gaussian model, on {picktwo} dimensions')
        fig1.savefig(savepth + '3D_gaussian.png', dpi=150)#dpi=150

        plt.clf()


    def plot_proj_1D(self, vectors, targets, savepth, istestset=True, acc=0.):
        print(f'now plotting 1D projection')
        if targets.shape[1] == 1: targets = self.from_one_col_to_binary(targets)
        cdict = {0: 'pink', 1: 'yellow'}
        indices_of_classes = [np.where(targets[:, i] == 1) for i in range(self.num_classes)] if istestset \
            else self.indices_of_classes_train
        data_proj = np.matmul(vectors, self.weight)  # e.g. in shape (n, 10)

        fig, ax = plt.subplots()
        for i in range(len(indices_of_classes)):
            proj = data_proj[indices_of_classes[i]]
            ys = np.random.normal(0, 0.01, len(proj))
            colorRGB = np.array(colors.to_rgb(cdict[i]))
            utils.imscatter(proj, ys, vectors[indices_of_classes[i]], ax, colorRGB)
            # ax.scatter(proj, ys, color=cdict[i], label=f'{self.digits_name[i]}')
        ax.axvline(x=self.threshold_proj, label='projection threshold', c="green")
        ax.legend()
        plt.title(f'{"Test" if istestset else "Train"} (acc={acc * 100: .1f}%) samples\' projection')
        plt.savefig(savepth, )#dpi=150
        plt.clf()

    def plot_proj_2D(self, vectors, targets, savepth, istestset=True, picktwo=None, acc=0.):
        print("now plotting 2D projection")
        if not picktwo: picktwo = [3, 6]
        cmap = cm.rainbow(np.linspace(0, 1, len(self.digits_name)))
        indices_of_classes = [np.where(targets[:, i] == 1) for i in range(self.num_classes)] if istestset \
            else self.indices_of_classes_train
        data_proj = np.matmul(vectors, self.weight)  # e.g. in shape (n, 10)

        fig, ax = plt.subplots()
        for i in range(len(indices_of_classes)):
            colorRGB = cmap[i][:3]
            proj = data_proj[indices_of_classes[i]]
            utils.imscatter(proj[:, picktwo[0]], proj[:, picktwo[1]], vectors[indices_of_classes[i]], ax, colorRGB)
        ax.legend()
        plt.title(f'{"Test" if istestset else "Train"} (acc={acc * 100: .1f}%) samples\' projection on {picktwo} dimension')
        plt.savefig(savepth,)# dpi=150
        plt.clf()

    def plot_weight(self, savepth):
        print("now plotting weight projection")
        if self.num_classes == 2:
            plt.imshow(self.weight.reshape((IMGSHAPE, IMGSHAPE)), cmap='gray')
            plt.title(f'Weight of LDA, binary on {self.digits_name}')
        else:
            for i in range(self.num_classes):
                plt.subplot(2, self.num_classes / 2, i + 1)
                weight = self.weight[:, i].reshape((IMGSHAPE, IMGSHAPE))
                _mean, _std = weight.mean(), weight.std()
                # print(weight.mean(), weight.std(), np.max(weight), np.min(weight))

                plt.title(i)
                plt.imshow(weight, cmap='gray', vmax=_mean+5*_std, vmin=_mean-5*_std)
                frame1 = plt.gca()
                frame1.axes.get_xaxis().set_visible(False)
                frame1.axes.get_yaxis().set_visible(False)
        plt.savefig(savepth, )#dpi=150
        plt.clf()



if __name__ == '__main__':

    '''
    randomly sample 10000 trains, 5000 tests, the LDA accuracy is not stable. train_acc for line 1, test_acc for line 2
    [0.8165, 0.82099999999999995, 0.82650000000000001, 0.82920000000000005, 0.79120000000000001, 0.79100000000000004, 0.80920000000000003, 0.8165, 0.82340000000000002, 0.82869999999999999, 0.81499999999999995, 0.83489999999999998, 0.82469999999999999, 0.81820000000000004, 0.81669999999999998, 0.82330000000000003, 0.81459999999999999, 0.81979999999999997, 0.78339999999999999, 0.83579999999999999, 0.81520000000000004, 0.80159999999999998, 0.82079999999999997, 0.81389999999999996, 0.83399999999999996, 0.82330000000000003, 0.8347, 0.81610000000000005, 0.80759999999999998, 0.81969999999999998, 0.82010000000000005, 0.78239999999999998, 0.84109999999999996, 0.78620000000000001, 0.83140000000000003, 0.8054, 0.8357, 0.81559999999999999, 0.81259999999999999, 0.81740000000000002, 0.80800000000000005, 0.8135, 0.80669999999999997, 0.78890000000000005, 0.82969999999999999, 0.82499999999999996, 0.81879999999999997, 0.82550000000000001, 0.83579999999999999, 0.82669999999999999, 0.79690000000000005, 0.81840000000000002, 0.7923, 0.78049999999999997, 0.81950000000000001, 0.82999999999999996, 0.81440000000000001, 0.82640000000000002, 0.81579999999999997, 0.82899999999999996, 0.79610000000000003, 0.82169999999999999, 0.82769999999999999, 0.81799999999999995, 0.82330000000000003, 0.81430000000000002, 0.80310000000000004, 0.82509999999999994, 0.82530000000000003, 0.82130000000000003, 0.82330000000000003, 0.80940000000000001, 0.81940000000000002, 0.82530000000000003, 0.83069999999999999, 0.81699999999999995, 0.81840000000000002, 0.80920000000000003, 0.82189999999999996, 0.80479999999999996, 0.82369999999999999, 0.8145, 0.82269999999999999, 0.82279999999999998, 0.81210000000000004, 0.80930000000000002, 0.8296, 0.81659999999999999, 0.82830000000000004, 0.80979999999999996, 0.80500000000000005, 0.82809999999999995, 0.81200000000000006, 0.81779999999999997, 0.82340000000000002, 0.81120000000000003, 0.83489999999999998, 0.80930000000000002, 0.7792, 0.82040000000000002]
    [0.56140000000000001, 0.51980000000000004, 0.46360000000000001, 0.57720000000000005, 0.52939999999999998, 0.43359999999999999, 0.58479999999999999, 0.4012, 0.27679999999999999, 0.4446, 0.41520000000000001, 0.65820000000000001, 0.36959999999999998, 0.53900000000000003, 0.43459999999999999, 0.50360000000000005, 0.50280000000000002, 0.37540000000000001, 0.40960000000000002, 0.52400000000000002, 0.43380000000000002, 0.31580000000000003, 0.215, 0.45700000000000002, 0.44979999999999998, 0.40479999999999999, 0.61419999999999997, 0.14419999999999999, 0.30780000000000002, 0.24060000000000001, 0.41980000000000001, 0.40760000000000002, 0.49619999999999997, 0.38219999999999998, 0.33900000000000002, 0.57599999999999996, 0.41199999999999998, 0.37740000000000001, 0.32640000000000002, 0.55320000000000003, 0.44600000000000001, 0.36820000000000003, 0.47839999999999999, 0.1938, 0.48099999999999998, 0.58120000000000005, 0.52300000000000002, 0.32940000000000003, 0.43859999999999999, 0.53420000000000001, 0.35360000000000003, 0.36180000000000001, 0.55200000000000005, 0.31780000000000003, 0.3498, 0.375, 0.45319999999999999, 0.24759999999999999, 0.4546, 0.4496, 0.48620000000000002, 0.46339999999999998, 0.26519999999999999, 0.38519999999999999, 0.36199999999999999, 0.3846, 0.55579999999999996, 0.46860000000000002, 0.32400000000000001, 0.35360000000000003, 0.3528, 0.4632, 0.37080000000000002, 0.35399999999999998, 0.26540000000000002, 0.44259999999999999, 0.4768, 0.62760000000000005, 0.61280000000000001, 0.29759999999999998, 0.22020000000000001, 0.44819999999999999, 0.39760000000000001, 0.48559999999999998, 0.42020000000000002, 0.3538, 0.44600000000000001, 0.32579999999999998, 0.5736, 0.40679999999999999, 0.29980000000000001, 0.45319999999999999, 0.30559999999999998, 0.46579999999999999, 0.39800000000000002, 0.38919999999999999, 0.43059999999999998, 0.5232, 0.38, 0.49880000000000002]
    '''

    ''' 
    estimate on two given digits (i, j) (j>i), the LDA accuracy is list[i][j]. train_acc for line 1, test_acc for line 2
    [[-1, 0.9947887879984209, 0.9874803149606299, 0.49284473488295144, 0.4879311784849244, 0.4813104856610139, 0.47808448716818064, 0.9991791841090043, 0.4829151535160119, 0.4958474576271186], [-1, 0.9947887879984209, 0.9874803149606299, 0.49284473488295144, 0.4879311784849244, 0.4813104856610139, 0.47808448716818064, 0.9991791841090043, 0.4829151535160119, 0.4958474576271186], [-1, 0.9947887879984209, 0.9874803149606299, 0.49284473488295144, 0.4879311784849244, 0.4813104856610139, 0.47808448716818064, 0.9991791841090043, 0.4829151535160119, 0.4958474576271186], [-1, 0.9947887879984209, 0.9874803149606299, 0.49284473488295144, 0.4879311784849244, 0.4813104856610139, 0.47808448716818064, 0.9991791841090043, 0.4829151535160119, 0.4958474576271186], [-1, 0.9947887879984209, 0.9874803149606299, 0.49284473488295144, 0.4879311784849244, 0.4813104856610139, 0.47808448716818064, 0.9991791841090043, 0.4829151535160119, 0.4958474576271186], [-1, 0.9947887879984209, 0.9874803149606299, 0.49284473488295144, 0.4879311784849244, 0.4813104856610139, 0.47808448716818064, 0.9991791841090043, 0.4829151535160119, 0.4958474576271186], [-1, 0.9947887879984209, 0.9874803149606299, 0.49284473488295144, 0.4879311784849244, 0.4813104856610139, 0.47808448716818064, 0.9991791841090043, 0.4829151535160119, 0.4958474576271186], [-1, 0.9947887879984209, 0.9874803149606299, 0.49284473488295144, 0.4879311784849244, 0.4813104856610139, 0.47808448716818064, 0.9991791841090043, 0.4829151535160119, 0.4958474576271186], [-1, 0.9947887879984209, 0.9874803149606299, 0.49284473488295144, 0.4879311784849244, 0.4813104856610139, 0.47808448716818064, 0.9991791841090043, 0.4829151535160119, 0.4958474576271186], [-1, 0.9947887879984209, 0.9874803149606299, 0.49284473488295144, 0.4879311784849244, 0.4813104856610139, 0.47808448716818064, 0.9991791841090043, 0.4829151535160119, 0.4958474576271186]]
    [[-1, 0.9933806146572104, 0.9875403784033225, 0.5053868756121449, 0.4929718875502008, 0.47545357524012805, 0.48162162162162164, 0.9949647532729103, 0.4865134865134865, 0.49067070095814425], [-1, 0.9933806146572104, 0.9875403784033225, 0.5053868756121449, 0.4929718875502008, 0.47545357524012805, 0.48162162162162164, 0.9949647532729103, 0.4865134865134865, 0.49067070095814425], [-1, 0.9933806146572104, 0.9875403784033225, 0.5053868756121449, 0.4929718875502008, 0.47545357524012805, 0.48162162162162164, 0.9949647532729103, 0.4865134865134865, 0.49067070095814425], [-1, 0.9933806146572104, 0.9875403784033225, 0.5053868756121449, 0.4929718875502008, 0.47545357524012805, 0.48162162162162164, 0.9949647532729103, 0.4865134865134865, 0.49067070095814425], [-1, 0.9933806146572104, 0.9875403784033225, 0.5053868756121449, 0.4929718875502008, 0.47545357524012805, 0.48162162162162164, 0.9949647532729103, 0.4865134865134865, 0.49067070095814425], [-1, 0.9933806146572104, 0.9875403784033225, 0.5053868756121449, 0.4929718875502008, 0.47545357524012805, 0.48162162162162164, 0.9949647532729103, 0.4865134865134865, 0.49067070095814425], [-1, 0.9933806146572104, 0.9875403784033225, 0.5053868756121449, 0.4929718875502008, 0.47545357524012805, 0.48162162162162164, 0.9949647532729103, 0.4865134865134865, 0.49067070095814425], [-1, 0.9933806146572104, 0.9875403784033225, 0.5053868756121449, 0.4929718875502008, 0.47545357524012805, 0.48162162162162164, 0.9949647532729103, 0.4865134865134865, 0.49067070095814425], [-1, 0.9933806146572104, 0.9875403784033225, 0.5053868756121449, 0.4929718875502008, 0.47545357524012805, 0.48162162162162164, 0.9949647532729103, 0.4865134865134865, 0.49067070095814425], [-1, 0.9933806146572104, 0.9875403784033225, 0.5053868756121449, 0.4929718875502008, 0.47545357524012805, 0.48162162162162164, 0.9949647532729103, 0.4865134865134865, 0.49067070095814425]]
    '''

    '''
    pick one digit i, and randomly sample the same length from the left digits, LDA accuracy is list[i]. train_acc for line 1, test_acc for line 2 
    [0.5, 0.9744141204390389, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    [0.5, 0.9788546255506608, 0.5, 0.5, 0.5005091649694501, 0.5016816143497758, 0.49947807933194155, 0.5, 0.49948665297741274, 0.5]
    '''

    parser = argparse.ArgumentParser(description='Logistic Regression')

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

    logdir = f'./LDAresults/sample_{sample_name_flag}'
    lda = LDA()

    if args.sample1vsrest or args.sample1vs1:
        tracc, teacc = lda.fit(train_images, train_labels, test_images, test_labels)
        lda.plot_weight(os.path.join(logdir, 'weight.png'))
        lda.plot_proj_1D(train_images, train_labels, savepth=os.path.join(logdir, f'1D_train.png'),
                         istestset=False, acc=float(tracc))
        lda.plot_proj_1D(test_images, test_labels, savepth=os.path.join(logdir, f'1D_test.png'),
                         istestset=True, acc=float(tracc))
    else:
        tracc, teacc = lda.fit(train_images, train_labels, test_images, test_labels)
        lda.plot_weight(os.path.join(logdir, 'weight.png'))
        i, j = 1, 6  # draw dimensions
        lda.plot_bivariate_gaussians(savepth=os.path.join(logdir, f'3D_gaussian_on_{i}_{j}'),
                                     picktwo=[i, j])  # or use None for default [3,6]
        lda.plot_proj_2D(train_images, train_labels, savepth=os.path.join(logdir, f'2D_on_{i}_{j}_train.png'),
                         istestset=False, picktwo=[i, j], acc=float(tracc))
        lda.plot_proj_2D(test_images, test_labels, savepth=os.path.join(logdir, f'2D_on_{i}_{j}_test.png'),
                         istestset=True, picktwo=[i, j], acc=float(teacc))
