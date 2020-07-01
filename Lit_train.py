from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from test_tube import HyperOptArgumentParser
from argparse import ArgumentParser
from lit_unet import Lightning_Unet
from data.const import COMPUTECANADA
import pathlib
import os
import torch


def main(hparams):
    """
    Trains the Lightning model as specified in `hparams`
    """

    model = Lightning_Unet(hparams)

    if COMPUTECANADA:
        default_root_dir = "/home/jueqi/projects/def-jlevman/U-Net_MRI-Data/log"
        checkpoint_file = "{pathlib.Path(__file__).resolve().parent}/log/checkpoint/{epoch}-{val_dice:.2f}"
    else:
        default_root_dir = "./log/checkpoint"
        checkpoint_file = "./log/checkpoint/{epoch}-{val_dice:.2f}"

    # After training finishes, use best_model_path to retrieve the path to the best
    # checkpoint file and best_model_score to retrieve its score.
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_file,
        save_top_k=3,
        verbose=True,
        monitor='val_dice',
        mode='max',
        prefix=''
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        strict=True,
        verbose=False,
        mode='min'
    )

    # model = LitUnet(args).load_from_checkpoint('./log/checkpoint')

    tb_logger = loggers.TensorBoardLogger(hparams.TensorBoardLogger)

    trainer = Trainer(
        gpus=hparams.gpus,
        # amp_level='O2', precision=16,
        # num_nodes=8, distributed_backend='ddp',
        check_val_every_n_epoch=1,
        # log every k batches instead
        row_log_interval=10,
        # set the interval at which you want to log using this trainer flag.
        log_save_interval=2,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        # runs 1 train, val, test  batch and program ends
        fast_dev_run=True,
        default_root_dir=default_root_dir,
        logger=tb_logger,
        max_epochs=10000,
        # resume_from_checkpoint='./log/checkpoint',
        profiler=True
    )

    trainer.fit(model)

    # (1) load the best checkpoint automatically (lightning tracks this for you)
    trainer.test()

    # (3) test using a specific checkpoint
    # trainer.test(ckpt_path='/path/to/my_checkpoint.ckpt')


# On Windows all of your multiprocessing-using code must be guarded by if __name__ == "__main__":
if __name__ == "__main__":
    parser = ArgumentParser(description='Trainer args', add_help=False)
    parser.add_argument("--gpus", type=int, default=1, help='which gpus')
    parser.add_argument("--TensorBoardLogger", dest='TensorBoardLogger', default='/home/jq/Desktop/log',
                        help='TensorBoardLogger dir')
    parser.add_argument("--name", dest='name', default="making train and val image in the same range, using dice loss")
    parser = Lightning_Unet.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)
