# Created by Haoyue Dai@06/27/2020

import numpy as np
import os
import utils
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import random
import argparse
from mpl_toolkits.mplot3d import axes3d, Axes3D
plt.rcParams.update({'figure.max_open_warning': 0})
IMGSHAPE = 28

def draw_64_images(vectors, pred_labels, gt_labels, savepth,):
    '''
    :param vectors: in shape (64, 784)
    :param pred_labels: in shape (64,)
    :param gt_labels: in shape (64,)
    :return:
    '''
    vectors = utils.un_normalize(vectors)
    vectors = vectors.reshape((-1, IMGSHAPE, IMGSHAPE))
    fig = plt.figure(figsize=(8, 8))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(64): # is it Ture:
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(vectors[i], cmap=plt.cm.binary, interpolation='nearest')
        if pred_labels[i] == gt_labels[i]:
            ax.text(0, 7, str(pred_labels[i]), color='green')
        else:
            ax.text(0, 7, str(pred_labels[i]), color='red')

    plt.savefig(savepth)
    plt.clf()


def draw_2dim_support_vecs(clf, vectors, labels, kernel, savepth):
    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    fig, ax = plt.subplots()
    X0, X1 = vectors[:, 0], vectors[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.rainbow, alpha=0.8)
    ax.scatter(X0, X1, c=labels, cmap=plt.cm.rainbow, s=10, edgecolors='k')
    ax.set_ylabel('PC2')
    ax.set_xlabel('PC1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(f'SVM-{kernel}: decison surface using the first 2 PCA projected features')
    ax.legend()
    plt.savefig(savepth)
    plt.clf()

    # from mlxtend.plotting import plot_decision_regions
    # plot_decision_regions(vectors, labels, clf=clf, legend=2)
    # plt.savefig(savepth)
    # plt.clf()

def SVM(train_images, train_labels, test_images, test_labels, max_iter=300, kernel='linear', C=1.0, gamma='auto', poly_degree=3, logdir='./results'):
    os.makedirs(logdir, exist_ok=True)
    label_to_index = lambda x: np.argmax(x, axis=1) if x.shape[1] > 1 else x.flatten()
    train_labels = label_to_index(train_labels)
    test_labels = label_to_index(test_labels)

    clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=poly_degree, max_iter=max_iter)
    clf.fit(train_images, train_labels)
    try:
        from joblib import dump, load
        dump(clf, os.path.join(logdir, 'SVM.joblib'))
    except: pass

    train_preds = clf.predict(train_images)
    train_accuracy = np.mean(train_preds == train_labels)

    test_preds = clf.predict(test_images)
    test_accuracy = np.mean(test_preds == test_labels)

    print(f'Train accuracy = {train_accuracy}. Test accuracy = {test_accuracy}')

    conf_mat = confusion_matrix(test_labels, test_preds)
    gt_num_of_class = np.sum(conf_mat, axis=1)

    conf_mat = conf_mat / gt_num_of_class[:, None] * 100.
    # normalize, now conf_mat[i, j] = #images_in_class_i_but_classified_to_j / # image_in_class_i %

    plt.figure(figsize=(10, 7)) #figsize=(10, 7)
    sn.heatmap(conf_mat, annot=True, fmt='.1f', cmap=plt.cm.Blues)
    plt.title(f'{kernel} SVM: confusion matrix on testset (%)')
    plt.ylabel('ground truth')
    plt.xlabel('prediction')
    plt.savefig(os.path.join(logdir, 'confusion_matrix.pdf'))
    plt.clf()

    support_vector_indices = clf.support_
    support_vectors_num_in_each_class = clf.n_support_

    if train_images.shape[1] == IMGSHAPE ** 2: # avoid drawing 2-dim PCA
        random64 = sorted(random.sample(support_vector_indices.tolist(), k=64))
        draw_64_images(train_images[random64],
                       train_preds[random64],
                       train_labels[random64],
                       savepth=os.path.join(logdir, 'support_vectors.pdf'))

    elif train_images.shape[1] == 2:
        draw_2dim_support_vecs(clf, train_images, train_labels, kernel, os.path.join(logdir, 'decision_boundary.pdf'))

    return train_accuracy, test_accuracy, support_vectors_num_in_each_class.tolist()

def pre_PCA(vectors, labels, savedir, n_components=2, classes_num=10, istestset=False):
    os.makedirs(savedir, exist_ok=True)
    nameflag = "testset" if istestset else "trainset"
    np.save(os.path.join(savedir, f'PCA_{n_components}_{nameflag}_label_onehot.npy'), labels)
    label_to_index = lambda x: np.argmax(x, axis=1) if x.shape[1] > 1 else x.flatten()
    labels = label_to_index(labels)
    pca = PCA(n_components=n_components)
    pca.fit(vectors)
    X_new = pca.transform(vectors)
    evr = pca.explained_variance_ratio_
    print(f'PCA completed, explained var ratio={evr}')
    np.save(os.path.join(savedir, f'PCA_{n_components}_{nameflag}_data.npy'), X_new)

    cmap = plt.cm.rainbow(np.linspace(0, 1, classes_num))
    fig, ax = plt.subplots()
    for i in range(classes_num):
        indices_i = np.where(labels == i)
        X_new_i = X_new[indices_i]
        colorRGB = cmap[i][:3]
        utils.imscatter(X_new_i[:, 0], X_new_i[:, 1], vectors[indices_i], ax, colorRGB)
    plt.title(f'PCA: first {n_components} dims on {nameflag}, var ratio={evr}')
    plt.savefig(os.path.join(savedir, f'PCA_{n_components}_{nameflag}.png'), dpi=150)
    plt.clf()

if __name__ == '__main__':
    '''
    on the same parameter C = 1.0; gamma = 'auto'; poly_degree = 3; train_num, test_num = 10000, 5000, different performances
    {'linear': {'train_accuracy': 1.0, 'test_accuracy': 0.9088, 'support_vectors_num_in_each_class': [191, 181, 301, 296, 268, 326, 209, 240, 351, 319]}, 'rbf': {'train_accuracy': 0.9846, 'test_accuracy': 0.9562, 'support_vectors_num_in_each_class': [276, 201, 416, 442, 404, 488, 338, 330, 484, 474]}, 'poly': {'train_accuracy': 0.9887, 'test_accuracy': 0.9566, 'support_vectors_num_in_each_class': [214, 200, 370, 383, 408, 486, 272, 333, 384, 438]}, 'sigmoid': {'train_accuracy': 0.8699, 'test_accuracy': 0.8668, 'support_vectors_num_in_each_class': [299, 235, 417, 475, 396, 518, 333, 339, 526, 522]}}
    
    full 60000, 10000:
    {'linear': {'train_accuracy': 0.9799166666666667, 'test_accuracy': 0.9311, 'support_vectors_num_in_each_class': [581, 494, 1133, 1291, 981, 1308, 697, 961, 1403, 1300]}, 'rbf': {'train_accuracy': 0.9899166666666667, 'test_accuracy': 0.9792, 'support_vectors_num_in_each_class': [750, 558, 1348, 1449, 1263, 1548, 927, 1149, 1666, 1679]}, 'poly': {'train_accuracy': 0.9925666666666667, 'test_accuracy': 0.9813, 'support_vectors_num_in_each_class': [585, 576, 1221, 1288, 1226, 1491, 806, 1129, 1320, 1498]}, 'sigmoid': {'train_accuracy': 0.8120333333333334, 'test_accuracy': 0.8219, 'support_vectors_num_in_each_class': [1051, 919, 1929, 2165, 1561, 2046, 1413, 1493, 2532, 2246]}}

    10000, 5000, PCA2:
    {'linear': {'train_accuracy': 0.18079999999999999, 'test_accuracy': 0.16719999999999999, 'support_vectors_num_in_each_class': [49, 50, 112, 115, 144, 149, 171, 223, 165, 226]}, 'rbf': {'train_accuracy': 0.23380000000000001, 'test_accuracy': 0.23799999999999999, 'support_vectors_num_in_each_class': [310, 175, 330, 273, 292, 356, 385, 281, 369, 261]}, 'poly': {'train_accuracy': 0.24610000000000001, 'test_accuracy': 0.24679999999999999, 'support_vectors_num_in_each_class': [18, 32, 53, 53, 48, 60, 71, 79, 82, 90]}, 'sigmoid': {'train_accuracy': 0.098400000000000001, 'test_accuracy': 0.1016, 'support_vectors_num_in_each_class': [212, 280, 283, 213, 293, 330, 340, 313, 316, 232]}}
    
    load 0 and 1, PCA2:
    {'linear': {'train_accuracy': 0.5950256612712199, 'test_accuracy': 0.5950256612712199, 'support_vectors_num_in_each_class': [32, 32]}, 'rbf': {'train_accuracy': 0.48101065929727593, 'test_accuracy': 0.48101065929727593, 'support_vectors_num_in_each_class': [50, 50]}, 'poly': {'train_accuracy': 0.9253849190682985, 'test_accuracy': 0.9253849190682985, 'support_vectors_num_in_each_class': [12, 15]}, 'sigmoid': {'train_accuracy': 0.7786814054480853, 'test_accuracy': 0.7786814054480853, 'support_vectors_num_in_each_class': [50, 50]}}

    load 0 and 1, no PCA2:
    {'linear': {'train_accuracy': 0.9999210422424003, 'test_accuracy': 0.99905437352245863, 'support_vectors_num_in_each_class': [36, 29]}, 'rbf': {'train_accuracy': 0.99984208448480061, 'test_accuracy': 0.99905437352245863, 'support_vectors_num_in_each_class': [47, 47]}, 'poly': {'train_accuracy': 0.82258191867350972, 'test_accuracy': 0.81607565011820327, 'support_vectors_num_in_each_class': [49, 49]}, 'sigmoid': {'train_accuracy': 0.99834188709040661, 'test_accuracy': 0.99810874704491725, 'support_vectors_num_in_each_class': [48, 48]}}

    load 0 and others, PCA2:
    {'linear': {'train_accuracy': 0.7603410433901739, 'test_accuracy': 0.7603410433901739, 'support_vectors_num_in_each_class': [31, 42]}, 'rbf': {'train_accuracy': 0.49924024987337495, 'test_accuracy': 0.49924024987337495, 'support_vectors_num_in_each_class': [50, 50]}, 'poly': {'train_accuracy': 0.6671450278575046, 'test_accuracy': 0.6671450278575046, 'support_vectors_num_in_each_class': [12, 12]}, 'sigmoid': {'train_accuracy': 0.5318250886375148, 'test_accuracy': 0.5318250886375148, 'support_vectors_num_in_each_class': [50, 50]}}

    load 0 and others, no PCA2:
    {'linear': {'train_accuracy': 0.88206989701164951, 'test_accuracy': 0.89591836734693875, 'support_vectors_num_in_each_class': [50, 45]}, 'rbf': {'train_accuracy': 0.97070741178456865, 'test_accuracy': 0.97091836734693882, 'support_vectors_num_in_each_class': [50, 50]}, 'poly': {'train_accuracy': 0.49957791659631945, 'test_accuracy': 0.50102040816326532, 'support_vectors_num_in_each_class': [50, 50]}, 'sigmoid': {'train_accuracy': 0.74506162417693733, 'test_accuracy': 0.73622448979591837, 'support_vectors_num_in_each_class': [50, 50]}}
    '''

    parser = argparse.ArgumentParser(description='Logistic Regression')

    parser.add_argument('--kernel', default='RBF', help='SVM kernels')
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

    logdir = f'./SVM/sample_{sample_name_flag}'
    train_accuracy, test_accuracy, support_vectors_num_in_each_class = SVM(train_images, train_labels, test_images,
                                                                           test_labels, kernel='RBF')

    # example on RBF kernel. For more attributes usage you can refer to codes above
    # confusion matrix, support vectors and decision boundaries are plotted
    # train_images, train_labels, test_images, test_labels = utils.load_MNIST_np(sample_train_test=(10000, 5000))
    # train_accuracy, test_accuracy, support_vectors_num_in_each_class = SVM(train_images, train_labels, test_images, test_labels, kernel='RBF')
