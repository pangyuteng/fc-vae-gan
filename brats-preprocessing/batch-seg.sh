#!/bin/bash

find /mnt/hd2/data/brats2019/MICCAI_BraTS_2019_Data_Training -name "*seg*" | xargs dirname > folders.txt

cat folders.txt | xargs -L 1 python segment_brain.py
