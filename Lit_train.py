from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from LitUnet import LitUnet
from argparse import ArgumentParser
import torch

# On Windows all of your multiprocessing-using code must be guarded by if __name__ == "__main__":
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--data_dir', type=str, help='data.tar folder')
    # add model specific args
    parser = LitUnet.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # After training finishes, use best_model_path to retrieve the path to the best
    # checkpoint file and best_model_score to retrieve its score.

    checkpoint_callback = ModelCheckpoint(
        filepath="log/checkpoint/{epoch}-{val_IoU:.2f}",
        save_top_k=1,
        verbose=True,
        monitor='val_IoU',
        mode='max',
        prefix=''
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        strict=True,
        verbose=False,
        mode='min'
    )

    model = LitUnet(args)
    # model = LitUnet(args).load_from_checkpoint('./log/checkpoint')

    trainer = Trainer.from_argparse_args(
        args,
        gpus=1,
        check_val_every_n_epoch=1,
        # log every k batches instead
        row_log_interval=100,
        # set the interval at which you want to log using this trainer flag.
        log_save_interval=10,
        # checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        # runs 1 train, val, test  batch and program ends
        # fast_dev_run=True,
        default_root_dir='log/checkpoint',
        profiler=True
    )

    trainer.fit(model)
