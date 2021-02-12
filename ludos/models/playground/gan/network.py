import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from box import Box
from pl_bolts.models.autoencoders.components import (resnet18_decoder,
                                                     resnet18_encoder)
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            *self.block(latent_dim, 128,
                        normalize=False), *self.block(128, 256),
            *self.block(256, 512), *self.block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))), nn.Tanh())

    def block(self, in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class BasicGAN(pl.LightningModule):
    """
    The generator G(z,\theta) generate samples the match the distribution
    of the data, while the discriminator, D(x;\theta_g) gives the prob
    that x came from the data rather than G.
    """
    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, Box):
            raise ValueError('Pass a dict instead')

        self.save_hyperparameters('cfg')
        self.cfg = Box(cfg)
        self.generator = Generator(
            img_shape=self.cfg.input.shape,
            latent_dim=self.cfg.network.latent_dim)
        self.discrimator = Discriminator(img_shape=self.cfg.input.shape)

        self.validation_z = torch.randn(8, self.cfg.network.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, ypred, ytrue):
        return F.binary_cross_entropy(ypred, ytrue)

    def configure_optimizers(self):
        params = self.cfg.network.generator.to_dict()
        opt_gen = torch.optim.Adam(self.generator.parameters(), **params)

        params = self.cfg.network.discriminator.to_dict()
        opt_disc = torch.optim.Adam(self.discrimator.parameters(), **params)
        return [opt_gen, opt_disc], []

    def training_step(self, batch, batch_idx, optimizer_idx):

        imgs, _ = batch

        z = torch.randn(imgs.shape[0], self.cfg.network.latent_dim)
        z = z.type_as(imgs)
        sample_imgs = self.generator(z)

        # sample z
        if optimizer_idx % 2 == 0:
            # train the generator to full the discrimator - the discriminator
            # should predict all ones - aka true label.
            # all fakes
            ytrue = torch.ones(imgs.size(0), 1).type_as(imgs)
            ypred = self.discrimator(sample_imgs)
            gen_loss = self.adversarial_loss(ypred, ytrue)
            self.log_dict({'gen_loss': gen_loss})
            return gen_loss

        # true images
        ytrue = torch.ones(imgs.size(0), 1).type_as(imgs)
        ypred = self.discrimator(imgs)
        real_loss = self.adversarial_loss(ypred, ytrue)

        # fake images
        ytrue = torch.zeros(imgs.size(0), 1).type_as(imgs)
        ypred = self.discrimator(sample_imgs.detach())
        fake_loss = self.adversarial_loss(ypred, ytrue)

        disc_loss = 0.5 * (real_loss + fake_loss)

        self.log_dict(
            {
                'real_loss': real_loss,
                'fake_loss': fake_loss,
                'disc_loss': disc_loss
            })

        return disc_loss

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        sample = self(z)
        grid = torchvision.utils.make_grid(sample)
        self.logger.experiment.add_image(
            'generated_image', grid, self.current_epoch)
