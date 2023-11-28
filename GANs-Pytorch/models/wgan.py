import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()

        self.img_shape = img_shape

        def block(in_features, out_features, norm=True):
            layers = [nn.Linear(in_features, out_features)]
            if norm:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.features = nn.Sequential(
            *block(latent_dim, 128, False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, x):
        image = self.features(x)
        image = image.view(image.shape[0], *self.img_shape)
        return image



class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):

        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity