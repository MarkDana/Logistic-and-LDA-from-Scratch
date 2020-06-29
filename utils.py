# Created by Haoyue Dai@06/27/2020

import gzip, os
import numpy as np
import torch
import torchvision.datasets as datasets
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import torchvision.transforms as transforms
import random


IMGCOL = 28
MEAN, STD = 0.1307, 0.3081
FILES = ['train-images-idx3-ubyte.gz',
         'train-labels-idx1-ubyte.gz',
         't10k-images-idx3-ubyte.gz',
         't10k-labels-idx1-ubyte.gz']
TRANS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,))
]),

def un_normalize(vectors):
    '''
    :param vectors: in shape (n, 784)
    :return: in shape (n, 784)
    '''
    return (vectors * STD) + MEAN


def load_MNIST_np(dir='./data/MNIST/raw', normlize=True, load_binary=None, load_two_classes=None, sample_train_test=None):
    if not os.path.exists(dir):
        print("Now downloading MNIST to ./data/")
        datasets.MNIST(root='./data/', train=True, transform=None, target_transform=None, download=True)
    def _images(path, normlize=True):
        with gzip.open(path) as f:
            pixels = np.frombuffer(f.read(), 'B', offset=16) # first 16 bytes are magic_number, n_imgs, n_rows, n_cols
        imgs = pixels.reshape(-1, IMGCOL ** 2).astype('float32') / 255
        if normlize:
            imgs = (imgs - MEAN) / STD
        return imgs

    def _labels(path):
        with gzip.open(path) as f:
            integer_labels = np.frombuffer(f.read(), 'B', offset=8) # first 8 bytes are magic_number, n_labels
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype='uint8')
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

    def _sample_binary(images, labels, load_binary):
        # where 1 for images in given class, 0 for random samples from rest, load_binary like 3
        labels = np.argmax(labels, axis=1) #onehot to index

        indices1 = (np.where(labels == load_binary)[0]).tolist()
        indices0 = random.sample(np.where(labels != load_binary)[0].tolist(), len(indices1))
        indices = [(x, 1) for x in indices1] + [(x, 0) for x in indices0]
        random.shuffle(indices)

        images = images[[x[0] for x in indices]]
        labels = np.array([x[1] for x in indices], dtype=int)
        labels = labels.reshape((labels.shape[0], -1))
        return images, labels

    def _sample_two(images, labels, load_two_classes):
        # load two given classes, load_two_classes like (3,5)
        labels = np.argmax(labels, axis=1)  # onehot to index
        indices = np.where((labels == load_two_classes[0]) | (labels == load_two_classes[1]))
        images = images[indices]
        labels = (labels[indices] == load_two_classes[1]).astype(int)
        labels = labels.reshape((labels.shape[0], -1))
        return images, labels

    train_images = _images(os.path.join(dir, FILES[0]), normlize)
    train_labels = _labels(os.path.join(dir, FILES[1]))
    test_images = _images(os.path.join(dir, FILES[2]), normlize)
    test_labels = _labels(os.path.join(dir, FILES[3]))

    if load_binary != None:
        train_images, train_labels = _sample_binary(train_images, train_labels, load_binary)
        test_images, test_labels = _sample_binary(test_images, test_labels, load_binary)

    if load_two_classes != None:
        train_images, train_labels = _sample_two(train_images, train_labels, load_two_classes)
        test_images, test_labels = _sample_two(test_images, test_labels, load_two_classes)

    if sample_train_test != None:
        indices_train = random.sample(list(range(len(train_labels))), sample_train_test[0])
        indices_test = random.sample(list(range(len(test_labels))), sample_train_test[1])
        train_images, train_labels, test_images, test_labels = train_images[indices_train], train_labels[indices_train], \
                                                               test_images[indices_test], test_labels[indices_test]

    return train_images, train_labels, test_images, test_labels


def drawCurveDonkey(intxtpath, outimgpath, title, xlabel='epoch', par1label='loss', par2label='accuracy(%)'):
    xs = []
    p1s = []
    p2s = []

    with open(intxtpath, 'r') as fin:
        lines = [l.strip() for l in fin.readlines()]
        for line in lines:
            x, p1, p2 = line.split('\t')
            xs.append(int(x))
            p1s.append(float(p1))
            p2s.append(float(p2))

    fig = plt.figure()
    host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])
    par1 = ParasiteAxes(host, sharex=host)
    host.parasites.append(par1)
    host.axis['right'].set_visible(False)
    par1.axis['right'].set_visible(True)
    par1.set_ylabel(par2label)
    par1.axis['right'].major_ticklabels.set_visible(True)
    par1.axis['right'].label.set_visible(True)
    fig.add_axes(host)
    host.set_xlabel(xlabel)
    host.set_ylabel(par1label)
    p1, = host.plot(np.array(xs), np.array(p1s), label=par1label)
    p2, = par1.plot(np.array(xs), np.array(p2s), label=par2label)
    plt.title(title)
    host.legend()
    host.axis['left'].label.set_color(p1.get_color())
    par1.axis['right'].label.set_color(p2.get_color())
    plt.savefig(outimgpath, dpi=150)
    plt.clf()

def imscatter(xs, ys, images, ax, colorRGB, zoom=0.5):
    artists = []
    images = un_normalize(images)

    RGBmean = np.mean(colorRGB)
    images = 1 - images  # pencil -> 0 (black), background -> 1 (white)
    images = images.reshape(-1, IMGCOL, IMGCOL)
    larger = np.where(images > RGBmean) # larger means more likely to be background, set color; if not, keep original
    images = np.repeat(images[:, :, :, None], 3, axis=3)
    images[larger] = colorRGB

    for i, (x0, y0) in enumerate(zip(xs, ys)):
        image = images[i]
        im = OffsetImage(image, zoom=zoom)  #[::2, ::2]
        ab = AnnotationBbox(im, (x0, y0), frameon=False) # xycoords='data', frameon=False
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([xs, ys]))
    ax.autoscale()
    return artists

