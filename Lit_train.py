from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
from lit_unet import Lightning_Unet
from pathlib import Path
from data.const import COMPUTECANADA
import pickle
import pathlib
import os
import torch


def main(hparams):
    """
    Trains the Lightning model as specified in `hparams`
    """
    model = Lightning_Unet(hparams)

    if COMPUTECANADA:
        cur_path = Path(__file__).resolve().parent.parent
        default_root_dir = cur_path / "log"
        checkpoint_file = Path(__file__).resolve().parent / "checkpoint/{epoch}-{val_dice:.2f}"
        if not os.path.exists(Path(__file__).resolve().parent / "checkpoint"):
            os.mkdir(Path(__file__).resolve().parent / "checkpoint")
    else:
        default_root_dir = "./log"
        checkpoint_file = "./log/checkpoint"
        if not os.path.exists(checkpoint_file):
            os.mkdir(checkpoint_file)
        checkpoint_file = Path(checkpoint_file) / "{epoch}-{val_dice:.2f}"

    if not os.path.exists(default_root_dir):
        os.mkdir(default_root_dir)

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
        num_nodes=hparams.nodes,
        distributed_backend='ddp',
        check_val_every_n_epoch=1,
        # log every k batches instead
        row_log_interval=10,
        # set the interval at which you want to log using this trainer flag.
        log_save_interval=2,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        # runs 1 train, val, test  batch and program ends
        fast_dev_run=False,
        default_root_dir=default_root_dir,
        logger=tb_logger,
        max_epochs=10000,
        # resume_from_checkpoint='./log/checkpoint',
        profiler=True,
        auto_lr_find=False,
        # simulate a larger batch size for gradient descent to provide a good estimate
        # accumulate_grad_batches=4,
    )

    # if COMPUTECANADA:
    #     pickle.dumps(model)
    # lr_finder = trainer.lr_find(model)
    #
    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    #
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print(f"recommend learning_rate: {new_lr}")
    # model.hparams.learning_rate = new_lr

    trainer.fit(model)
    # (1) load the best checkpoint automatically (lightning tracks this for you)
    trainer.test()

    # (3) test using a specific checkpoint
    # trainer.test(ckpt_path='/path/to/my_checkpoint.ckpt')


# On Windows all of your multiprocessing-using code must be guarded by if __name__ == "__main__":
if __name__ == "__main__":
    parser = ArgumentParser(description='Trainer args', add_help=False)
    parser.add_argument("--gpus", type=int, default=1, help='how many gpus')
    parser.add_argument("--nodes", type=int, default=1, help='how many nodes')
    parser.add_argument("--TensorBoardLogger", dest='TensorBoardLogger', default='/home/jq/Desktop/log',
                        help='TensorBoardLogger dir')
    parser.add_argument("--name", dest='name', default="using cropped data")
    parser = Lightning_Unet.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)
