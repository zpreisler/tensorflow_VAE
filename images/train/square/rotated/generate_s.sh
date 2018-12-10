#!/bin/bash

for input in `ls ss1*.png`
do
	for angle in `seq 3 3 45`
	do
		output=$(echo $input | sed "s/.png/_rot_$angle.png/")
		echo $angle $input $output
		convert -rotate $angle -crop "640x640+0+0" $input $output
	done
done
