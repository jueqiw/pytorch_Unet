import argparse


class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PyTorch')
        self.parser.add_argument('--data_dir', type=str, default='', help='the path of the dataset')

    def parse(self):
        opt = self.parser.parse_args()
        return opt