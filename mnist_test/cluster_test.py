from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

from traincluster1 import load_checkpoint
from torchvision.datasets import MNIST
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import MultipleLocator

def fetch_mnist_image(label):
    """Return a random image from the MNIST dataset with label.

    @param label: integer
                  a integer from 0 to 9
    @return: torch.autograd.Variable
             MNIST image
    """
    mnist_dataset = datasets.MNIST('./data', train=False, download=True,
                                   transform=transforms.ToTensor())
    images = mnist_dataset.test_data.numpy()
    labels = mnist_dataset.test_labels.numpy()
    images = images[labels == label]
    image = images[np.random.choice(np.arange(images.shape[0]))]
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)
    return Variable(image, volatile=True)


def fetch_mnist_text(label):
    """Randomly generate a number from 0 to 9.

    @param label: integer
                  a integer from 0 to 9
    @return: torch.autograd.Variable
             Variable wrapped around an integer.
    """
    text = torch.LongTensor([label])
    return Variable(text, volatile=True)

def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    #for x, y, s in zip(X, Y, labels):
        #c = cm.rainbow(int(255 * s / 9))
        #plt.text(x, y, s,  backgroundcolor=c, fontsize=9)
    x_major = MultipleLocator(10)
    y_major = MultipleLocator(10)
    ax =plt.gca()
    ax.xaxis.set_major_locator(x_major)
    ax.yaxis.set_major_locator(y_major)
    plt.scatter(X,Y,25,labels,cmap=plt.cm.Spectral)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.show()
    plt.pause(0.01)

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=64,
                        help='Number of images and texts to sample [default: 64]')
    # condition sampling on a particular images
    parser.add_argument('--condition-on-image', type=int, default=None,
                        help='If True, generate text conditioned on an image.')
    # condition sampling on a particular text
    parser.add_argument('--condition-on-text', type=int, default=None,
                        help='If True, generate images conditioned on a text.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint("./cluster_first_models/model_best.pth.tar", use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()

    test_loader = torch.utils.data.DataLoader(
        MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=200, shuffle=False)
    k=1
    plt.ion()
    for batch_idx, (image, text) in enumerate(test_loader):
        if args.cuda:
            image = image.cuda()
            text = text.cuda()

        image = Variable(image, volatile=True)
        text = Variable(text, volatile=True)
        batch_size = len(image)

        img_recon, txt_recon,h = model(image, text) #[100,784]
        #img_recon, txt_recon,  mu, logvar = model(image, text)
        img_recon = F.sigmoid(img_recon).cpu().data
        txt_recon = F.log_softmax(txt_recon, dim=1).cpu().data
        '''
        print(img_recon.shape)
        # save image samples to filesystem
        save_image(img_recon.view(args.n_samples, 1, 28, 28),
                   './sample_image3.png')
        # save text samples to filesystem
        with open('./sample_text3.txt', 'w') as fp:
            txt_recon_np = txt_recon.detach().numpy()
            txt_recon_np = np.argmax(txt_recon_np, axis=1).tolist()
            for i, item in enumerate(txt_recon_np):
                fp.write('Text (%d): %s\n' % (i, item))
        '''
        #'''
        h = Variable(h.cpu()).detach().numpy()
        text = Variable(text.cpu()).detach().numpy()
        #print(text)

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 200
        low_dim_embs = tsne.fit_transform(h[:plot_only, :])
        labels = text[:plot_only]
        plot_with_labels(low_dim_embs, labels)
        if k ==1:
            break
        k = k + 1

    plt.ioff()