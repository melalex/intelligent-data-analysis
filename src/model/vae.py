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


class DeepVAE(nn.Module):

    def __init__(self, latent_dim=20):
        super(DeepVAE, self).__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        encoder_output_len = 4 * 4 * 512

        self.fc_mu = nn.Linear(encoder_output_len, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_len, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoder_output_len),
            nn.Unflatten(1, (512, 4, 4)),
            # Reverse Block 4
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            # Reverse Block 3
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            # Reverse Block 2
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            # Reverse Block 1
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
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
