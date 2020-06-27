#!/bin/bash
#SBATCH --time=00-02:00      # time (DD-HH:MM)
#SBATCH --mem=1G   # memory
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send email
#SBATCH --mail-type=ALL

module load python/3.6
source ~/ENV/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"

echo -e '\n'
cd $SLURM_TMPDIR
mkdir work
# --strip-components prevents making double parent directory
echo "$(date +"%T"):  Copying data"
tar -xf /project/6005889/U-Net_MRI-Data/data.tar -C work && echo "$(date +"%T"):  Copied data"
# Now do my computations here on the local disk using the contents of the extracted archive...

## The computations are done, so cSlean up the data set.S..

python /home/jueqi/projects/def-jlevman/jueqi/Unet1/get_all_readable_file.py

echo "$(date +"%T"):  taring data"
cd work
tar -cf /project/6005889/U-Net_MRI-Data/crop_data.tar img/ label/
echo "$(date +"%T"):  tared data"