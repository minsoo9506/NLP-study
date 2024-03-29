from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _train(self, x, y, config):
        self.model.train()

        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

        total_loss = 0.0

        for i, (x_i, y_i) in enumerate(zip(x,y)):
            y_hat_i = model(x_i)
            loss_i = crit(y_i, y_hat_i.squeeze())

            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()

            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        self.model.eval()

        with torch.no_grad():
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x,y)):
                y_hat_i = model(x_i)
                loss_i = crit(y_i, y_hat_i.squeeze())

                total_loss += float(loss_i)

        return total_loss / len(x)

    def train(self, train_data, validate_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            validate_loss = self._validate(validate_data[0], validate_data[1], config)

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print(f"Epoch({epoch_index+1}/{config.n_epochs}): train_loss={train_loss:.4f} valid_loss={validate_loss:.4f} lowest_loss={lowest_loss:.4f}")

        self.model.load_state_dict(best_model)




