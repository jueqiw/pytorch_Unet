"""Code is from https://github.com/CBICA/BrainMaGe/blob/64a2e3a50a0565372a25cd5090e12ddca5af8f6b/BrainMaGe/utils/optimizers.py#L13
"""

import torch.optim as optim
import sys


def fetch_optimizer(optimizer, lr, model):
    # Setting up the optimizer
    if optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=float(lr),
                              momentum=0.9, nesterov=True)
    elif optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=float(lr),
                               betas=(0.9, 0.999),
                               weight_decay=0.00005)
    else:
        print("Sorry, {} is not supported or some sort of spell error. Please\
               choose from the given options!".format(optimizer))
        sys.stdout.flush()
        sys.exit(0)
    return optimizer