import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_loaders
import pytorch_lightning as pl
from MyLightning import NN

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--max_epochs', type=int, default=20)

    config = p.parse_args()

    return config

def main(config):
    train_loader, valid_loader, test_loader = get_loaders(config)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    model = NN(784, 64, 10)
    trainer = pl.Trainer(max_epochs=config.max_epochs, gpus=1)
    trainer.fit(model,train_loader,valid_loader)

if __name__ == '__main__':
    config = define_argparser()
    main(config)