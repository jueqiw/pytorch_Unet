git init
git add ./*
git rm ./summary/Unet/*
git commit -m "try use the probability as label,outputing is two class and using binary_cross_entropy, and compute the dice and iou"
git push -u pytorchstripping master