import torch
from tqdm.notebook import tqdm
import torch.nn.functional as F


class VaeTrainer:

    def __init__(self, model, epochs, optimizer):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer

    def vae_loss(self, x_reconstructed, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL loss
        return recon_loss + kl_div

    def train(self, x):
        with tqdm(total=self.epochs) as p_bar:
            for _ in range(self.epochs):
                self.optimizer.zero_grad()
                x_reconstructed, mu, logvar = self.model(x)
                loss = self.vae_loss(x_reconstructed, x, mu, logvar)
                loss.backward()
                self.optimizer.step()

                progress_postfix = {
                    "loss": loss.item(),
                }

                p_bar.set_postfix(**progress_postfix)
                p_bar.update()
