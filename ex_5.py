import sys

import ex4
import numpy as np
from gcommand_dataset import GCommandLoader
import torch.utils.data
from torch import optim
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F



def main():
    train_set=GCommandLoader('gcommands/train')
    test_set = GCommandLoader('gcommands/test')
    valid_set = GCommandLoader('gcommands/valid')
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=10, shuffle=True,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=10, shuffle=True,
        pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=10, shuffle=True,
        pin_memory=True)

if __name__ == "__main__":
    main()