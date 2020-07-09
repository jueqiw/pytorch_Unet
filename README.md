# pytorch_Unet
- [X] add cropping
- [X] try Leaky ReLU with negative_slope = <img src="https://render.githubusercontent.com/render/math?math=1e^{-2}">
- [X] test cropping
- [X] add parallel to cropping (not working)
- [X] rewrite the compute loss part (not working)
- [x] change the save file name to original name
- [X] write test part, using not interpreted image
- [X] rewrite part do not use temp disk
- [X] try other server
- [X] install model
- [X] test crop or not crop's dice
- [X] modify learning rate
- [X] read the plot image with maximum loss
- [X] visualize the cropped image
- [X] bid the image with the filename
- [X] rebuild the data.tar file
- [X] using regularization
- [X] fix plot image part
- [X] test by 1 run, and let every model run
- [X] try another two cluster
### fine tune part
- [ ] writing fine-tune part 
- [X] try the Instance Normal instead of GN
- [ ] try dice and BCE with weight
- [X] try ResUnet
- [ ] try denseUnet
- [ ] change the deepth of unet
- [X] downsampling part
- [X] change the kernal size

### data prepare part
- [ ] do nothing, using interplot
- [ ] do cropping and patch-based
- [ ] do resample, padding and interplot

- [ ] reading nnUnet again and using padding and interplot instead of interplotting
- [ ] read about nnUnet learning rate
- [ ] try dilated convolution

## only test on the CC359 dataset

| model|loss|activation|epoch| dice | sensitivity | specificity |
|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|
| 1| dice |ReLU |23|**96.57%**|**97.02%**|99.23%|
| 2| BCE |ReLU|21|95.48%|95.98%|**99.47%**|
| 3| dice|leaky ReLU|**12**|95.61%|98.54%|98.84%|

## test on the whole dataset ()

| model|loss|activation|epoch| dice | sensitivity | specificity |
|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|
| 1| dice |ReLU |23|**96.57%**|**97.02%**|99.23%|
| 2| BCE |ReLU|21|95.48%|95.98%|**99.47%**|
| 3| dice|leaky ReLU|**12**|95.61%|98.54%|98.84%|
