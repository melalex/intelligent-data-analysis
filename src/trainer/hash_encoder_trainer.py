import torch
import torch.nn.functional as F

from tqdm.notebook import tqdm


class HashEncoderTrainer:

    def __init__(self, model, epochs, optimizer):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer

    def hash_loss(self, x_reconstructed, x, z_binary):
        recon_loss = F.mse_loss(x_reconstructed, x)
        binarization_loss = torch.mean(torch.abs(torch.abs(z_binary) - 1))
        return recon_loss + 0.01 * binarization_loss

    def train(self, x):
        with tqdm(total=self.epochs) as p_bar:
            for _ in range(self.epochs):
                self.optimizer.zero_grad()
                encoded, decoded = self.model(x)
                loss = self.hash_loss(decoded, x, encoded)
                loss.backward()
                self.optimizer.step()

                progress_postfix = {
                    "loss": loss.item(),
                }

                p_bar.set_postfix(**progress_postfix)
                p_bar.update()
