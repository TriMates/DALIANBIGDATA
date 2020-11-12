# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 08:31:38 2020

@author: Administrator
"""
import sys
import numpy as np

import torch
from torch.autograd import Variable

from torchvision.datasets import MNIST
from utils.print_result import print_result
import math
from sklearn.utils import shuffle
import torch.nn.functional as F
from torchvision import datasets, transforms
from clustermvae1 import MAEGenerator, ImageDiscrimintor, TextDiscrimintor

def load_checkpoint(file_path, use_cuda=False):
     checkpoint = torch.load(file_path) if use_cuda else \
         torch.load(file_path, map_location=lambda storage, location: storage)
     # Todo: 修改mvae
     model = MAEGenerator(checkpoint['n_latents'])
     model.load_state_dict(checkpoint['state_dict'])
     return model

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
    # data set
    test_loader = torch.utils.data.DataLoader(
        MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=100, shuffle=False)
    n_clusters = 10
    batch_size = 100

    H = np.random.uniform(0, 1, [10000, 64])
    gt = np.random.uniform(0, 1, [10000])
    for batch_idx, (image, text) in enumerate(test_loader):
        start_idx, end_idx = batch_idx * batch_size, (batch_idx + 1) * batch_size
        image = Variable(image, volatile=True)
        text = Variable(text, volatile=True)
        batch_size = len(image)

        img_recon, txt_recon,h = model(image, text) #[100,784]
        img_recon = F.sigmoid(img_recon).cpu().data
        txt_recon = F.log_softmax(txt_recon, dim=1).cpu().data

        h = Variable(h.cpu()).detach().numpy()

        H[start_idx: end_idx, ...] = h
        gt[start_idx: end_idx, ...] = text.detach().numpy()

    print_result(n_clusters, H, gt)
