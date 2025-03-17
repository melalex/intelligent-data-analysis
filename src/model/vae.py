import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Unflatten(1, (16, 4, 4)),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ConvTranspose2d(16, 6, kernel_size=5, padding=0),
            nn.ConvTranspose2d(6, 6, kernel_size=2, stride=2),
            nn.ConvTranspose2d(6, 1, kernel_size=5, padding=0),
        )

    def encode(self, x):
        h = F.relu(self.encoder(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
