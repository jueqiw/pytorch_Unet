from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from test_tube import HyperOptArgumentParser
from argparse import ArgumentParser
from lit_unet import Lightning_Unet
from data.const import COMPUTECANADA
from data.const import get_data_dir
import os
import torch


# export
def main(hparams):
    """
    Trains the Lightning model as specified in `hparams`
    """

    get_data_dir(hparams.data_dir)
    model = Lightning_Unet(hparams)

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
        min_delta=0.00,
        patience=5,
        strict=True,
        verbose=False,
        mode='min'
    )

    # model = LitUnet(args).load_from_checkpoint('./log/checkpoint')

    if COMPUTECANADA:
        default_root_dir = "/home/jueqi/projects/def-jlevman/U-Net_MRI-Data/log"
    else:
        default_root_dir = "./log/checkpoint"

    tb_logger = loggers.TensorBoardLogger(hparams.TensorBoardLogger)

    trainer = Trainer(
        default_save_path=default_root_dir,
        # Enable 16-bit
        # faster for training also without loss in performance
        amp_level='O1',
        precision=16,
        gpus=hparams.gpus,
        # num_nodes=4, distributed_backend='ddp',
        check_val_every_n_epoch=1,
        # log every k batches instead
        row_log_interval=100,
        # set the interval at which you want to log using this trainer flag.
        log_save_interval=10,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        # runs 1 train, val, test  batch and program ends
        fast_dev_run=True,
        default_root_dir=default_root_dir,
        logger=tb_logger,
        max_nb_epochs=10000,
        # resume_from_checkpoint='./log/checkpoint',
        profiler=True
    )

    trainer.fit(model)


# On Windows all of your multiprocessing-using code must be guarded by if __name__ == "__main__":
if __name__ == "__main__":
    parser = ArgumentParser(description='Trainer args', add_help=False)
    parser.add_argument("--data_dir", type=str, default="", help='data.tar folder', dest="data_dir")
    parser.add_argument("--gpus", type=int, default=1, help='which gpus')
    parser.add_argument("--TensorBoardLogger", dest='TensorBoardLogger', default='/home/jq/Desktop/log',
                        help='TensorBoardLogger dir')
    parser.add_argument("--name", dest='name', default="Using original model, resize the picture to predict")
    parser = Lightning_Unet.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)
