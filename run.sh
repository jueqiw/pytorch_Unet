#!/bin/bash
#SBATCH --gres=gpu:t4:1  # request GPU "generic resource"
#SBATCH --cpus-per-task=16   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=150G   # memory
#SBATCH --output=out-%j.out  # %N for node name, %j for jobID
#SBATCH --time=03-00:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send email
#SBATCH --mail-type=ALL

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
echo "$(date +"%T"):  Activated python virtualenv"
pip3 install --no-index torch nibabel  && echo "$(date +"%T"):  Installed nibabel, torch"
pip3 install --no-index torchio && echo "$(date +"%T"):  Installed torchio from local wheel!"
pip3 install --no-index scikit-image && echo "$(date +"%T"):  Installed torchio from local wheel!"

echo -e '\n\n\n'
cd $SLURM_TMPDIR
mkdir work
# --strip-components prevents making double parent directory
echo "$(date +"%T"):  Copying data"
tar -xf /home/jueqi/projects/def-jlevman/jueqi/my_data.tar -C work && echo "$(date +"%T"):  Copied data"
# Now do my computations here on the local disk using the contents of the extracted archive...
cd work
## The computations are done, so clean up the data set...
#tar -cf ~/projects/def-foo/johndoe/results.tar work

EPOCHS=500
BATCH_SIZE=4

# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  Executing train.py"
python3 /home/jueqi/projects/def-jlevman/jueqi/pytorch_Unet/train.py \
       --data_dir="$SLURM_TMPDIR" \
       --epochs="$EPOCHS" \
       --batch-size=$BATCH_SIZE \
       && echo "$(date +"%T"):  Successfully executed train.py"