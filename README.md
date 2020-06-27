# pytorch_Unet
- [X] add cropping
- [X] try Leaky ReLU with negative_slope = <img src="https://render.githubusercontent.com/render/math?math=1e^{-2}">
- [X] test cropping
- [ ] add parallel to cropping
- [ ] rebuild the data.tar file
- [ ] rewrite the compute loss part
- [ ] change the kernal size
- [ ] try the Instance Normal instead of GN
- [ ] try denseUnet
- [ ] try ResUnet
- [ ] try dilated convolution

## only test on the CC359 dataset

| model|loss|activation|epoch| dice | sensitivity | specificity |
|-----|:----:|:-----:|:----:|:-----:|:----:|:-----:|
| 1| dice |ReLU |23|**96.57%**|**97.02%**|99.23%|
| 2| BCE |ReLU|21|95.48%|95.98%|99.47%|
| 3| dice|leaky ReLU|**12**|95.61%|98.54%|98.84%|
