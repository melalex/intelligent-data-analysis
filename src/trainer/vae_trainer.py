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
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def train_with_loader(self, dataloader, device):
        dataloader_len = len(dataloader)
        iter_count = dataloader_len * self.epochs
        with tqdm(total=iter_count) as p_bar:
            for epoch in range(self.epochs):

                total_loss = 0
                for i, images in enumerate(dataloader):
                    images = images.to(device)
                    self.optimizer.zero_grad()
                    x_reconstructed, mu, logvar = self.model(images)
                    loss = self.vae_loss(x_reconstructed, images, mu, logvar)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                    progress_postfix = {
                        "loss": total_loss / (i + 1),
                    }

                    p_bar.set_postfix(**progress_postfix)
                    p_bar.update()
                
                print(
                    f"Epoch {epoch + 1} of {self.epochs} finished. Loss is {total_loss / dataloader_len}."
                )
