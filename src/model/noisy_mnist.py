from torch import nn
import torch

from src.model.auto_encoder import NoiseCancelingAutoEncoder


class NoisyMnist(nn.Module):

    def __init__(self, autoencoder: NoiseCancelingAutoEncoder, num_classes: int):
        super(NoisyMnist, self).__init__()
        self.encoder = autoencoder.encoder
        self.fc = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.encoder(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
