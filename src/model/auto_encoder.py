from torch import nn
import torch


class Lab3AutoEncoder(nn.Module):

    def __init__(self, input_dim, encoding_dim):
        super(Lab3AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Lab3LinearAutoEncoder(nn.Module):

    def __init__(self, input_dim, encoding_dim):
        super(Lab3LinearAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Linear(1024, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 1024),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoEncoder(nn.Module):

    activations = {
        "relu": nn.ReLU(),
        "linear": nn.Identity(),
    }

    def __init__(self, units, activation="linear"):
        super(AutoEncoder, self).__init__()

        encoder_units = []
        decoder_units = []

        for it, next in zip(units[:-2], units[1:-1]):
            encoder_units.append(nn.Linear(it, next))
            encoder_units.append(self.activations[activation])

        encoder_units.append(nn.Linear(units[-2], units[-1]))

        self.encoder = nn.Sequential(*encoder_units)

        decoder_units = []

        for it, next in reversed(list(zip(units[2:], units[1:-1]))):
            decoder_units.append(nn.Linear(it, next))
            decoder_units.append(self.activations[activation])

        decoder_units.append(nn.Linear(units[1], units[0]))

        self.decoder = nn.Sequential(*decoder_units)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class HashEncoder(nn.Module):

    def __init__(self, units):
        super(HashEncoder, self).__init__()

        encoder_units = []
        decoder_units = []

        for it, next in zip(units[:-2], units[1:-1]):
            encoder_units.append(nn.Linear(it, next))
            encoder_units.append(nn.ReLU())

        encoder_units.append(nn.Linear(units[-2], units[-1]))

        self.encoder = nn.Sequential(*encoder_units)

        decoder_units = []

        for it, next in reversed(list(zip(units[2:], units[1:-1]))):
            decoder_units.append(nn.Linear(it, next))
            decoder_units.append(nn.ReLU())

        decoder_units.append(nn.Linear(units[1], units[0]))

        self.decoder = nn.Sequential(*decoder_units)

    def forward(self, x):
        encoded = self.encoder(x)
        binary_encoding = torch.tanh(encoded)
        decoded = self.decoder(binary_encoding)
        return binary_encoding, decoded


class ConvHashEncoder(nn.Module):

    def __init__(self):
        super(ConvHashEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 4, 4)),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ConvTranspose2d(16, 6, kernel_size=5, padding=0),
            nn.ConvTranspose2d(6, 6, kernel_size=2, stride=2),
            nn.ConvTranspose2d(6, 1, kernel_size=5, padding=0),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        binary_encoding = torch.tanh(encoded)
        decoded = self.decoder(binary_encoding)
        return binary_encoding, decoded


class NoiseCancelingAutoEncoder(nn.Module):

    def __init__(self):
        super(NoiseCancelingAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ConvTranspose2d(16, 6, kernel_size=5, padding=0),
            nn.ConvTranspose2d(6, 6, kernel_size=2, stride=2),
            nn.ConvTranspose2d(6, 1, kernel_size=5, padding=0),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
