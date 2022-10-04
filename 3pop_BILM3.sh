#!/bin/bash


for i in 35 85 110 120
do
	for nr in {4..5}
	do
	echo $i
		python train_meso.py --end_timepoint=$i --train_Nrinit=$nr
	done
done

