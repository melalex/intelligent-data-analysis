from tqdm.notebook import tqdm


class AutoEncoderTrainer:

    def __init__(self, model, loss_fun, epochs, optimizer):
        self.model = model
        self.loss_fun = loss_fun
        self.epochs = epochs
        self.optimizer = optimizer

    def train(self, x):
        with tqdm(total=self.epochs) as p_bar:
            for _ in range(self.epochs):
                self.optimizer.zero_grad()
                _, decoded = self.model(x)
                loss = self.loss_fun(decoded, x)
                loss.backward()
                self.optimizer.step()

                progress_postfix = {
                    "loss": loss.item(),
                }

                p_bar.set_postfix(**progress_postfix)
                p_bar.update()
