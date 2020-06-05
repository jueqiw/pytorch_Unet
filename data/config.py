import argparse


class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PyTorch Unet brain stripping')
        self.parser.add_argument('--data_dir', type=str, default='', help='the path of the dataset')
        self.parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                                 help='Number of epochs', dest='epochs')
        self.parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                                 help='Batch size', dest='batchsize')
        self.parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                                 help='Learning rate', dest='lr')
        self.parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                                 help='Load model from a .pth file')
        # self.parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
        #                          help='Downscaling factor of the images')
        # self.parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
        #                          help='Percent of the data that is used as validation (0-100)')

    def parse(self):
        opt = self.parser.parse_args()
        return opt
