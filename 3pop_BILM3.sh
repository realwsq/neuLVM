#!/bin/bash


for i in 35 85 120
do
	for nr in {6..8}
	do
	echo $i
		python train_meso.py --end_timepoint=$i --train_Nrinit=$nr
	done
done

