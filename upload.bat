git init
git add ./*
git rm ./summary/Unet/*
git rm ./upload.bat
git commit -m "using padding and cropping to make Unet can put in different image size"
git push -u pytorchstripping master -f