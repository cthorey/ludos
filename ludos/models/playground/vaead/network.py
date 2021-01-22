from box import Box

import pytorch_lightning as pl
import torch
from pl_bolts.models.autoencoders.components import (resnet18_decoder,
                                                     resnet18_encoder)
from torch import nn


class VAE(pl.LightningModule):
    """
    A vaiational encoder --> objective
    """
    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, Box):
            raise ValueError('Pass a dict instead')

        self.save_hyperparameters('cfg')
        self.cfg = Box(cfg)
        self.learning_rate = self.cfg.network.lr

        # encoder: aka encode the inputs
        self.encoder = resnet18_encoder(False, False)

        # decoder: aka decode the latent variable
        # Not that the decoder do not generate the sample
        # It generates the parameters of the distributions
        # we use to generate the sample. - this is p_theta(x|z)
        self.decoder = resnet18_decoder(
            latent_dim=self.cfg.network.latent_dim,
            input_height=self.cfg.network.input_height,
            first_conv=False,
            maxpool1=False)

        # Now we need to generate the distributions parameters
        self.fc_mu = nn.Linear(self.cfg.network.enc_out_dim,
                               self.cfg.network.latent_dim)
        self.fc_var = nn.Linear(self.cfg.network.enc_out_dim,
                                self.cfg.network.latent_dim)

        # Additioal parameter for the variance of p_theta from
        # the encoder
        self.log_p_xz_std = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        params = {'lr': self.cfg.network.lr}
        optimizer = torch.optim.Adam(self.parameters(), **dict(params))
        return optimizer

    def training_step(self, batch, batch_idx):
        # encode the batch to the embedding
        x, _ = batch
        feat = self.encoder(x)  # Bx512

        # produce the parameters of the distribution q_phi(z|x)
        # mu,sigma**2
        mu, var = self.fc_mu(feat), self.fc_var(feat)  # Bx256,Bx256,
        std = torch.exp(0.5 * var)  # Bx256,

        # Sample from q_zx which we chose to be a multivariate
        # gaussian. However we need to be carefull.
        # Indeed the sampling is a stochastic process which we can't sample
        # from, however - it is possible to express z as a determistic
        # variable. z = mu + sigma x epsilon where epsilon ~ N(0,1)
        epsilon = torch.distributions.Normal(torch.zeros_like(mu),
                                             torch.zeros_like(std))
        z = mu + epsilon.sample() * var  # Bx256

        # Alternatively, in pytorch - we can use rsample which allow
        # to follow the path derivative
        q_zx = torch.distributions.Normal(mu, std)
        z = q_zx.rsample()  # Bx256

        # Let's focus on the reconstrution loss for now log(p_theta(x|z)).
        # We use the decoder to generate the parameters of p_theta(x|z)
        # We will assume our decoder is normal with unit variance
        p_xz_mu = self.decoder(z)
        # Add a new parameter  for the standard deviation - initial at one.
        scale = torch.exp(self.log_p_xz_std)
        p_xz = torch.distributions.Normal(p_xz_mu, scale)
        reconstruction_loss = p_xz.log_prob(x).sum(axis=(1, 2, 3))

        # Then now we need to compute the KL divergence
        # To do so, we need two distributions q and p.
        # q we have already ;)
        # for p we are using a multivariate normal
        # with zero mean and unit variance
        p = torch.distributions.Normal(torch.zeros_like(mu),
                                       torch.ones_like(std))
        log_pz = p.log_prob(z)
        log_qzx = q_zx.log_prob(z)

        kl = log_qzx - log_pz
        # this bit is a bit tricky. Since these are log probabilities
        # we can sum all the individual dimensions
        # to give us the multi-dimensional  probability
        kl = kl.sum(-1)

        # finally the loss is
        elbo = (-reconstruction_loss + kl)
        loss = elbo.mean()
        self.log_dict({
            'loss': loss,
            'elbo': elbo.mean(),
            'kl': kl.mean(),
            'recon_loss': reconstruction_loss.mean()
        })

        return loss
