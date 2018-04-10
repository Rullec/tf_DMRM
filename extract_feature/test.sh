for i in `ls ../utils/checkpoints/`  
do
	echo $i
	python test.py $i 1>>logout 2>logerr 
done
