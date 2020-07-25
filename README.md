### fine tune part
- [X] fix kernal size = 3 problem
- [X] writing fine-tune part
- [X] read the pytorch lightning pruning part
- [ ] find out why some value just won't work??? (cannot learn)
- [ ] try the resample, padding and interplot method
- [ ] adding data augmentation fine tune
- [ ] adding one more layer without concat
- [ ] try dice and BCE with weight
- [ ] try denseUnet
- [ ] Be more careful with the learning rate

### later to do 
- [X] write the predict part, and using the mp4 part
- [X] fix mp4 problem
- [X] update the mp4 quality
- [X] when test, the label do not do the data augmentation resize part
- [ ] do nothing, using interplot
- [ ] do cropping and patch-based
- [ ] write test part on the 1069 data
- [ ] make your own color bar

### Reading
- [ ] reading nnUnet again and using padding and interplot instead of interplotting
- [ ] read about nnUnet learning rate

## only test on the CC359 dataset

| model|loss|activation|epoch| dice | sensitivity | specificity |
|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|
| 1| dice |ReLU |23|**96.57%**|**97.02%**|99.23%|
| 2| BCE |ReLU|21|95.48%|95.98%|**99.47%**|
| 3| dice|leaky ReLU|**12**|95.61%|98.54%|98.84%|

