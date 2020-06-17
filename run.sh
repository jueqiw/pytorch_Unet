#!/bin/bash
#SBATCH --gres=gpu:t4:4  # request GPU "generic resource"
#SBATCH --nodes=4
#SBATCH --cpus-per-task=14   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=170G   # memory
#SBATCH --output=try1-%j.out  # %N for node name, %j for jobID
#SBATCH --time=00-02:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send email
#SBATCH --mail-type=ALL

module load python/3.6
source ~/ENV/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"

echo -e '\n\n\n'
cd $SLURM_TMPDIR
mkdir work
# --strip-components prevents making double parent directory
echo "$(date +"%T"):  Copying data"
tar -xf /project/6005889/U-Net_MRI-Data/data.tar -C work && echo "$(date +"%T"):  Copied data"
# Now do my computations here on the local disk using the contents of the extracted archive...
cd work
## The computations are done, so clean up the data set...
#tar -cf ~/projects/def-foo/johndoe/results.tar work

RUN=1
BATCH_SIZE=4
GPUS=4
LOG_DIR=/home/$USER/projects/def-jlevman/U-Net_MRI-Data/log
echo "$SLURM_TMPDIR"


# run script
echo -e '\n\n\n'
tensorboard --logdir="$LOG_DIR" --host 0.0.0.0 & python3 /home/$USER/projects/def-jlevman/jueqi/Unet1/Lit_train.py \
       --data_dir="$SLURM_TMPDIR" \
       --gpus="$GPUS" \
       --batch-size=$BATCH_SIZE \
       --run=$RUN \
       --name="Using original model, resize the picture to predict" \
       --TensorBoardLogger="$LOG_DIR"


#python3 /home/jueqi/projects/def-jlevman/jueqi/pytorch_Unet/data/const.py