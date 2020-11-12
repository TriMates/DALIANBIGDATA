from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
# from torchvision import datasets
from torchvision.utils import save_image
import numpy as np

from clustermvae1 import MAEGenerator, ImageDiscrimintor, TextDiscrimintor


def elbo_loss(recon_image, image, recon_text, text,
              lambda_image=1.0, lambda_text=1.0):
    """Bimodal ELBO loss function.

    @param recon_image: torch.Tensor
                        reconstructed image
    @param image: torch.Tensor
                  input image
    @param recon_text: torch.Tensor
                       reconstructed text probabilities
    @param text: torch.Tensor
                 input text (one-hot)
    @param mu: torch.Tensor
               mean of latent distribution
    @param logvar: torch.Tensor
                   log-variance of latent distribution
    @param lambda_image: float [default: 1.0]
                         weight for image BCE
    @param lambda_text: float [default: 1.0]
                       weight for text BCE
    @param annealing_factor: integer [default: 1]
                             multiplier for KL divergence term
    @return ELBO: torch.Tensor
                  evidence lower bound
    """
    image_bce, text_bce = 0, 0  # default params
    if recon_image is not None and image is not None:
        image_bce = torch.sum(binary_cross_entropy_with_logits(
            recon_image.view(-1, 1 * 28 * 28),
            image.view(-1, 1 * 28 * 28)), dim=1)

    if recon_text is not None and text is not None:
        text_bce = torch.sum(cross_entropy(recon_text, text), dim=1)

    ELBO = torch.mean(lambda_image * image_bce + lambda_text * text_bce)
    return ELBO


def binary_cross_entropy_with_logits(input, target):
    """Sigmoid Activation + Binary Cross Entropy

    @param input: torch.Tensor (size N)
    @param target: torch.Tensor (size N)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return (torch.clamp(input, 0) - input * target
            + torch.log(1 + torch.exp(-torch.abs(input))))


def cross_entropy(input, target, eps=1e-6):
    """k-Class Cross Entropy (Log Softmax + Log Loss)

    @param input: torch.Tensor (size N x K)
    @param target: torch.Tensor (size N x K)
    @param eps: error to add (default: 1e-6)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size(0) == input.size(0)):
        raise ValueError(
            "Target size ({}) must be the same as input size ({})".format(
                target.size(0), input.size(0)))

    log_input = F.log_softmax(input + eps, dim=1)
    y_onehot = Variable(log_input.data.new(log_input.size()).zero_())
    y_onehot = y_onehot.scatter(1, target.unsqueeze(1), 1)
    loss = y_onehot * log_input
    return -loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
     checkpoint = torch.load(file_path) if use_cuda else \
         torch.load(file_path, map_location=lambda storage, location: storage)
     # Todo: 修改mvae
     model = MAEGenerator(checkpoint['n_latents'])
     model.load_state_dict(checkpoint['state_dict'])
     return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-latents', type=int, default=64,
                        help='size of the latent embedding [default: 64]')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--annealing-epochs', type=int, default=200, metavar='N',
                        help='number of epochs to anneal KL for [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--lambda-image', type=float, default=1.,
                        help='multipler for image reconstruction [default: 1]')
    parser.add_argument('--lambda-text', type=float, default=10.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: False]')
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.isdir('./cluster_first_models'):
        os.makedirs('./cluster_first_models')

    dataloader = torch.utils.data.DataLoader(
        MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    generator = MAEGenerator(args.n_latents)
    text_discrimintor = TextDiscrimintor()
    img_discrimintor = ImageDiscrimintor()
    adversarial_loss = torch.nn.BCELoss()

    if args.cuda:
        generator.cuda()
        text_discrimintor.cuda()
        img_discrimintor.cuda()

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_TD = optim.Adam(text_discrimintor.parameters(), lr=args.lr)
    optimizer_ID = optim.Adam(img_discrimintor.parameters(), lr=args.lr)

    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    H = []
    for i in range(len(dataloader)):
        h = np.random.rand(args.batch_size, 10)
        u, s, vh = np.linalg.svd(h, full_matrices=False)
        W = u @ vh
        W = torch.from_numpy(W)
        H.append(W)

    best_loss = sys.maxsize
    # Traning
    for epoch in range(args.n_epochs):
        train_loss = 0
        for i, (imgs, text) in enumerate(dataloader):
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            valid = valid.cuda()
            fake = fake.cuda()
            imgs = imgs.cuda()
            text = text.cuda()

            real_imgs = Variable(imgs.type(Tensor))

            y_onehot = torch.zeros(text.size(0), 10).cuda()
            real_text = y_onehot.scatter(1, text.unsqueeze(1), 1)
            real_text = Variable(real_text.type(Tensor))  # torch.Size([100])
            W = H[i].cuda()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            gen_img, gen_text, h = generator(imgs, text)  # [batch_size, 784] [batch_size, 10]
            g_loss2 = adversarial_loss(text_discrimintor(gen_text), valid)  # [100, 1], [100, 1]
            joint_loss = elbo_loss(gen_img, imgs, gen_text, text, lambda_image=args.lambda_image,
                                   lambda_text=args.lambda_text)
            gen_img1 = F.sigmoid(gen_img)
            T = torch.t(h)  # h is [64,100] m*n
            WTW = torch.matmul(h, T)  # HT * H
            FTWTWF = torch.matmul(torch.matmul(torch.t(W.double()), WTW.double()), W.double())
            loss_k_means = torch.trace(WTW) - torch.trace(FTWTWF)
            g_loss1 = adversarial_loss(img_discrimintor(gen_img1), valid)
            g_loss = g_loss1 + g_loss2 + joint_loss + 1e-2 / 2 * loss_k_means
            train_loss += g_loss.item()
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_TD.zero_grad()
            text_real_loss = adversarial_loss(text_discrimintor(real_text), valid)
            text_fake_loss = adversarial_loss(text_discrimintor(gen_text.detach()), fake)
            d_loss_t = (text_real_loss + text_fake_loss) / 2
            d_loss_t.backward()
            optimizer_TD.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_ID.zero_grad()
            img_real_loss = adversarial_loss(img_discrimintor(real_imgs), valid)
            img_fake_loss = adversarial_loss(img_discrimintor(gen_img1.detach()), fake)
            d_loss_i = (img_real_loss + img_fake_loss) / 2
            d_loss_i.backward()
            optimizer_ID.step()

            if epoch % 50 == 0 and epoch != 0:
                hidden_val = np.asarray(h.cpu().detach())
                K = hidden_val.T  #H
                U, sigma, VT = np.linalg.svd(K)
                sorted_indices = np.argsort(sigma)
                topk_evecs = VT[sorted_indices[:-10 - 1:-1], :]
                F_new = topk_evecs.T
                H[i] = torch.from_numpy(F_new)

            if i % 20 == 1:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D_T loss: %f] [D_I loss: %f] [G loss: %f]"
                    % (epoch, args.n_epochs, i, len(dataloader), d_loss_t.item(), d_loss_i.item(), g_loss.item())
                )


        train_loss = train_loss / len(dataloader)
        is_best = train_loss < best_loss
        best_loss = min(train_loss, best_loss)
        # save the best model and current model
        save_checkpoint({
            'state_dict': generator.state_dict(),
            'best_loss': best_loss,
            'n_latents': args.n_latents,
            'optimizer': optimizer_G.state_dict(),
        }, is_best, folder='./cluster_first_models')