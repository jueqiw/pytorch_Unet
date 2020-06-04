#!/bin/bash
#SBATCH --gres=gpu:t4:1  # request GPU "generic resource"
#SBATCH --cpus-per-task=8   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=125G   # memory
#SBATCH --output=out-path.out  # %N for node name, %j for jobID
#SBATCH --time=00-03:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send email
#SBATCH --mail-type=ALL

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
echo "$(date +"%T"):  Activated python virtualenv"
pip install --no-index torch nibabel  && echo "$(date +"%T"):  Installed nibabel, torch"
pip install --no-index torchio && echo "$(date +"%T"):  Installed torchio from local wheel!"
pip install --no-index scikit-image && echo "$(date +"%T"):  Installed torchio from local wheel!"

for ((i=0; i<$(tput cols); i++)); do echo -e "=\c" ;done
cd $SLURM_TMPDIR
mkdir work
# --strip-components prevents making double parent directory
echo "$(date +"%T"):  Copying data"
tar -xf /home/jueqi/projects/def-jlevman/jueqi/my_data.tar -C work --strip-components 1 && echo "$(date +"%T"):  Copied data"
# Now do my computations here on the local disk using the contents of the extracted archive...
echo "print cur path:" && pwd
## The computations are done, so clean up the data set...
#tar -cf ~/projects/def-foo/johndoe/results.tar work

# run script
for ((i=0; i<$(tput cols); i++)); do echo -e "=\c" ;done
echo "$(date +"%T"):  Executing torch_test.py"
python /home/jueqi/projects/def-jlevman/jueqi/pytorch_Unet/train.py \
       --data_dir="$SLURM_TMPDIR" && echo "$(date +"%T"):  Successfully executed train.py"