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
- [ ] rebuild the data.tar file
- [ ] modify learning rate
- [ ] read about nnUnet learning rate
- [ ] try the Instance Normal instead of GN
- [ ] change the kernal size
- [ ] try dilated convolution
- [ ] try dice and BCE with weight
- [ ] try denseUnet
- [ ] try ResUnet
- [ ] rewrite the interprate part

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
