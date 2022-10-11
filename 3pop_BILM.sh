#!/bin/bash


for i in {0..4}
do
	echo $i
	python train_meso.py --end_timepoint=11 --train_Nrinit=$i
done
# end_timepoint >= time_end + a_cutoff = 10+1 = 11


# for i in 44 65 85 100 135 140 150 175 195 200
# do
# 	echo $i
# 	python train_meso.py --end_timepoint=$i --train_Nrinit=1
# done
