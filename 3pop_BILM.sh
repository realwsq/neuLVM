#!/bin/bash


for i in {1..5}
do
	echo $i
	python train_meso.py --end_timepoint=10 --train_Nrinit=$i
done

# for i in 44 65 85 100 135 140 150 175 195 200
# do
# 	echo $i
# 	python train_meso.py --end_timepoint=$i --train_Nrinit=1
# done
