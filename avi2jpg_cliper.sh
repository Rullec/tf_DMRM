#!/bin/bash
# put this script in target folder filled with .avi files, it can help you clip videos to imgs!
file=`ls -l | grep '.avi' | awk '{print $9}'`

for i in $file
do
	str=$i
	dirname=${str:0:20}
	echo $dirname
	mkdir $dirname
	ffmpeg -i $i ./$dirname/%03d.jpg -hide_banner 
	#echo $i processing succ!
done
