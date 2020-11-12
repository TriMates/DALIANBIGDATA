from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class ImageDiscrimintor(nn.Module):
    def __init__(self):
        super(ImageDiscrimintor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        img_validity = self.model(img_flat)
        return img_validity

class TextDiscrimintor(nn.Module):
    def __init__(self):
        super(TextDiscrimintor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, text):
        text_validity = self.model(text)
        return text_validity  # [batch_size * hidden]=[100 * 1]

class MAEGenerator(nn.Module):

    def __init__(self, n_latents):
        super(MAEGenerator, self).__init__()
        self.image_encoder = ImageEncoder(n_latents)
        self.image_decoder = ImageDecoder(2*n_latents)
        self.text_encoder  = TextEncoder(n_latents)
        self.text_decoder  = TextDecoder(2*n_latents)
        self.fuse_encodr = FuseEncoder(n_latents)
        self.n_latents = n_latents


    def forward(self, image=None, text=None):
        hidden_image = self.image_encoder(image)
        hidden_text = self.text_encoder(text)
        z = torch.cat((hidden_image, hidden_text), dim=1)
        h, z_ = self.fuse_encodr(z)
        img_recon  = self.image_decoder(z_)
        txt_recon  = self.text_decoder(z_)
        return img_recon, txt_recon, h  # [batch_size, 784], [batch_size, 10]

class ImageEncoder(nn.Module):

    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))
        return h

class ImageDecoder(nn.Module):

    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.fc1 = nn.Linear(n_latents, 512)
        self.fc2 = nn.Linear(512, 784)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        return self.fc2(h)  # NOTE: no sigmoid here. See train.py



class TextEncoder(nn.Module):

    def __init__(self, n_latents):
        super(TextEncoder, self).__init__()
        self.fc1   = nn.Embedding(10, 512)
        self.fc2   = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        return h

class TextDecoder(nn.Module):

    def __init__(self, n_latents):
        super(TextDecoder, self).__init__()
        self.fc1   = nn.Linear(n_latents, 512)
        self.fc2   = nn.Linear(512, 10)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        return self.fc2(h)  # NOTE: no softmax here. See train.py


class FuseEncoder(nn.Module):

    def __init__(self, n_latents):
        super(FuseEncoder, self).__init__()
        self.fce = nn.Linear(2*n_latents, n_latents)
        self.fcd = nn.Linear(n_latents, 2*n_latents)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fce(z))
        z_ = self.swish(self.fcd(h))
        return h, z_

class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)