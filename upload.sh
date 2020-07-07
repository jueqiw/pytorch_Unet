#!/bin/bash
git init
git add ./*
git rm ./log/
git rm ./exclude-file.txt
git rm __pycache__/
git rm ./data/__pycache__/
git rm ./model/__pycache__/
git rm ./postprocess/__pycache__/
git rm  ./utils/__pycache__/
git rm ./upload.sh
git rm ./upload.bat
git commit -m "adding test part"
git push -u pytorchstripping master -f