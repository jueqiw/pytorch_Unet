#!/bin/bash
#SBATCH --time=02-00:00      # time (DD-HH:MM)
#SBATCH --mem=260G   # memory
#SBATCH --account=def-jlevman
#SBATCH --output=crop-%j.out  # %N for node name, %j for jobID
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send email
#SBATCH --mail-type=ALL

module load python/3.6
source ~/ENV/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"

echo -e '\n'
#cd $SLURM_TMPDIR
#mkdir work
## --strip-components prevents making double parent directory
#echo "$(date +"%T"):  Copying data"
#tar -xf /project/6005889/U-Net_MRI-Data/readable_data.tar -C work && echo "$(date +"%T"):  Copied data"
# Now do my computations here on the local disk using the contents of the extracted archive...

echo "$(data +"%T"): starting ..."
python3 /home/jueqi/projects/def-jlevman/jueqi/Unet1/cropping.py && echo "$(date +"%T"):  Successfully!"
#echo "$(date +"%T"):  taring data"
#cd work
### The computations are done, so clean up the data set...
#tar -cf /project/6005889/U-Net_MRI-Data/cropped_data.tar img/ label/
tar -cf /project/6005889/U-Net_MRI-Data/cropped_data.tar /project/6005889/U-Net_MRI-Data/img/ /project/6005889/U-Net_MRI-Data/label/
echo "$(date +"%T"):  tared data"