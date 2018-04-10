#!/bin/bash
ObjStr=$1
keyword="\.py \.cpp \.sh"
blackword="\.npy \.h5 \.hdf5 \.jpg \.avi"
blackdir="video_data"
enable_keyword="1"
enable_blackword="0"

GlobalJudge=0

if [ $enable_keyword = "1" -a $enable_blackword = "1" ]; then
	echo 'you set both enable_keyword=1 and enable_blackword=1, there is a conflit.'
	exit
fi

dirrule()
{
	dirname=$1
	for tmp3 in $blackdir
	do
		judge3=`echo $dirname| grep $tmp3 | wc -l`
		if [ $judge3 -eq 1 ];then
			GlobalJudge=1
			return 1
		fi
	done
	GlobalJudge=0
}

filerule(){
	# jump it return 1
	# search it return 0
	
	filename=$1
	if [ "1" = $enable_keyword ];then
		for tmp1 in $keyword
		do
			judge1=`echo $filename| grep $tmp1 | wc -l`
			if [ $judge1 -eq 1 ];then
				GlobalJudge=0
				return 0
			fi
		done
		GlobalJudge=1
		return 1
	elif [ $enable_blackword = "1" ];then
		for tmp2 in $blackword
		do
			judge2=`echo $filename| grep $tmp2 | wc -l`
			if [ $judge2  -eq 1 ];then
				GlobalJudge=1
				return 1
			fi
		done
		GlobalJudge=0
	fi
	
}

findObj(){
	# find Obj in directory recursively
	echo working in `pwd`
	dir=`ls -l | grep '^d'|awk '{print $9}'`
	for i in $dir
	do
		dirrule $i
		tmp=$GlobalJudge
		res=`echo $tmp | grep 1 | wc -l`
		if [ $res -eq 1 ];then
			echo skip directory $i
			continue
		fi
		cd $i
		findObj
		cd ..
	done
	
	# find Obj in files
	file=`ls -l | grep ^- | awk '{print $9}'`
	for i in $file
	do	
		filerule $i
		# judge if the filename is in blacklist
		tmp=$GlobalJudge
		res=`echo $tmp | grep 1 | wc -l`
		if [ $res -eq 1 ];then
			echo skip `pwd`/$i
			continue
		fi
		
		num=`cat $i | grep $ObjStr |wc -l`
		if [ 0 -lt $num ]
		then
			echo successfully found $ObjStr in `pwd`/$i
		fi
	done
}

findObj 
