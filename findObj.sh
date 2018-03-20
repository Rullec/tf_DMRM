#!/bin/bash
#tips
echo this script is used to find someting in a directort or file content.
echo so we need 2 parameters: the first is search file, the second is objString.

prefile=$1
str=$2

findunit(){
	# $1: directory
	echo ????`pwd`
	nowdir=`ls -d /*`
	nowfile=`ls -l | grep ^-| awk '{print $9 }' | grep $str`
	for i in $nowfile:
	do
		echo $i
		res=`cat $i | grep $str |wc -l`
		if [ $res -eq 0 ]
		then
			echo `pwd`/$i
		fi
	done

	for i in $nowdir:
	do
		cd $i
		findunit $i
		cd ..
	done
	sleep 1
}

cd $1
findunit .

echo find $str over
