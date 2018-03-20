#!/bin/bash
ObjStr=$1

findObj(){
	# find Obj in directory recursively
	echo working in `pwd`
	dir=`ls -l | grep '^d'|awk '{print $9}'`
	for i in $dir
	do
		cd $i
		findObj
		cd ..
	done
	
	# find Obj in files
	file=`ls -l | grep ^- | awk '{print $9}'`
	for i in $file
	do	
		num=`cat $i | grep $ObjStr |wc -l`
		if [ 0 -lt $num ]
		then
			echo successfully found $ObjStr in `pwd`/$i
		fi
	done
}

findObj
