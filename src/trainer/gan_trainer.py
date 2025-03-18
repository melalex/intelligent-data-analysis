import torch
import torch.nn as nn
import torchvision.utils as vutils


class GanTrainer:

    def __init__(
        self,
        epochs,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        criterion,
        device
    ):
        self.epochs = epochs
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.criterion = criterion
        self.device = device

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self, dataloader, nz):
        real_label = 1.
        fake_label = 0.

        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        for epoch in range(self.epochs):
            for i, data in enumerate(dataloader, 0):

                self.discriminator.zero_grad()
                real_cpu = data[:1].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full(
                    (b_size,), real_label, dtype=torch.float, device=self.device
                )
                output = self.discriminator(real_cpu).view(-1)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                fake = self.generator(noise)
                label.fill_(fake_label)
                output = self.discriminator(fake.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.discriminator_optimizer.step()

                self.generator.zero_grad()
                label.fill_(real_label)
                output = self.discriminator(fake).view(-1)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.generator_optimizer.step()

                if i % 50 == 0:
                    print(
                        "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                        % (
                            epoch,
                            self.epochs,
                            i,
                            len(dataloader),
                            errD.item(),
                            errG.item(),
                            D_x,
                            D_G_z1,
                            D_G_z2,
                        )
                    )
