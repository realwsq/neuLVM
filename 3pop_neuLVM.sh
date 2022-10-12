#!/bin/bash


# We do five fits with different initialized parameters,
# the one with the lowest loss is considered as the final fitted model.
for i in {0..4}
do
	echo $i
	python train_meso.py --end_timepoint=11 --train_Noinit=$i
done
# end_timepoint >= time_end + a_cutoff = 10+1 = 11

