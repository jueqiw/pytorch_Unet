from pytorch_lightning import Trainer, loggers, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from argparse import ArgumentParser
from lit_unet import Lightning_Unet
from data.const import COMPUTECANADA
import shutil
import pickle
import pathlib
import os
import torch

DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def objective(trial):
# def main():
    parser = ArgumentParser(description='Trainer args', add_help=False)
    parser.add_argument("--gpus", type=int, default=1, help='which gpus')
    parser.add_argument("--TensorBoardLogger", dest='TensorBoardLogger', default='/home/jq/Desktop/log',
                        help='TensorBoardLogger dir')
    parser.add_argument("--name", dest='name', default="only using one dataset, uncropped data")
    parser.add_argument("--pruning", "-p", action="store_true",
                         help="Activate the pruning feature. `MedianPruner` stops unpromising "
                              "trials at the early stages of training.")
    parser = Lightning_Unet.add_model_specific_args(parser)
    hparams = parser.parse_args()

    default_root_dir = "./log"
    checkpoint_file = "./log/checkpoint/{epoch}-{val_dice:.2f}"

    if not os.path.exists(default_root_dir):
        os.mkdir(default_root_dir)
    if not os.path.exists("./log/checkpoint"):
        os.mkdir("./log/checkpoint")

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

    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=3,
    #     strict=True,
    #     verbose=False,
    #     mode='min'
    # )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We don't use any logger here as it requires us to implement several abstract
    # methods. Instead we setup a simple callback, that saves metrics from each validation step.
    tb_logger = loggers.TensorBoardLogger(hparams.TensorBoardLogger)

    metrics_callback = MetricsCallback()
    trainer = Trainer(
        gpus=hparams.gpus,
        # amp_level='O2', precision=16,
        # num_nodes=2, distributed_backend='ddp',
        check_val_every_n_epoch=1,
        # log every k batches instead
        row_log_interval=10,
        # set the interval at which you want to log using this trainer flag.
        log_save_interval=2,
        # checkpoint_callback=checkpoint_callback,
        # early_stop_callback=early_stop_callback,
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_dice"),
        # runs 1 train, val, test  batch and program ends
        fast_dev_run=False,
        default_root_dir=default_root_dir,
        logger=False,
        # logger=tb_logger,
        max_epochs=10000,
        # resume_from_checkpoint='./log/checkpoint',
        profiler=True,
        # auto_lr_find=True,
        # max_epochs=EPOCHS,
        callbacks=[metrics_callback],
    )
    # model = Lightning_Unet(hparams, trial)
    model = Lightning_Unet(hparams)
    # model = LitUnet(args).load_from_checkpoint('./log/checkpoint')

    # Run learning rate finder
    # trainer = Trainer()
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

    # if COMPUTECANADA:
    #     pickle.dumps(model)
    trainer.fit(model)

    # (1) load the best checkpoint automatically (lightning tracks this for you)
    # trainer.test()
    # (3) test using a specific checkpoint
    # trainer.test(ckpt_path='/path/to/my_checkpoint.ckpt')

    return metrics_callback.metrics[-1]["val_dice"].item()


# On Windows all of your multiprocessing-using code must be guarded by if __name__ == "__main__":
if __name__ == "__main__":
    # Pruner using the median stopping rule.
    # Prune if the trial’s best intermediate result is worse than median of intermediate
    # results of previous trials at the same step.
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=1000, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # offers a number of high-level operations on files and collections of files. In particular,
    # functions are provided which support file copying and removal. For operations on individual
    # files, see also the os module.
    shutil.rmtree(MODEL_DIR)
