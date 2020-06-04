#!/bin/bash
#SBATCH --gres=gpu:p100:1  # request GPU "generic resource"
#SBATCH --cpus-per-task=16   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --output=out-path.out  # %N for node name, %j for jobID
#SBATCH --time=00-03:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send email
#SBATCH --mail-type=ALL

module load cuda cudnn
source /home/jueqi/tensorflow/bin/activate

cd $SLURM_TMPDIR
pwd
mkdir work
cd work
# --strip-components prevents making double parent directory
echo "$(date +"%T"):  Copying data"
tar -xf /home/jueqi/projects/def-jlevman/jueqi/my_data.tar -C work --strip-components 1 && echo "$(date +"%T"):  Copied data"
# Now do my computations here on the local disk using the contents of the extracted archive...

tree

## The computations are done, so clean up the data set...
#cd $SLURM_TMPDIR
#tar -cf ~/projects/def-foo/johndoe/results.tar work

# run script
echo "$(date +"%T"):  Executing torch_test.py"
python /home/jueqi/projects/def-jlevman/jueqi/pytorch_Unet/train.py && echo "$(date +"%T"):  Successfully executed train.py"