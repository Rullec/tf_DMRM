from shutil import copy
import os
import sys
import random
reload(sys)
sys.setdefaultencoding('utf-8')

def get_dirinfo( path):
	filelist = []
	dirlist = []
	# if this path exists and it is a dir(not a file)
	if (os.path.exists(path) and os.path.isdir(path)):
		files = os.listdir(path)
		for file in files:
			m = os.path.join(path, file)  
			# m is dir or file?
			if (os.path.isdir(m)):    
				dirlist.append(m)
			else:
				filelist.append(m)
	else:
		raise('the directory ' + path + ' is not exist!')
	filelist.sort()
	dirlist.sort()
	return dirlist, filelist

# mova image to train_image
test_split = 0.05
val_split = 0.05

#clean workspace
if os.path.exists('./train_image/train.txt'):
	os.remove('./train_image/train.txt')
if os.path.exists('./val_image/val.txt'):
	os.remove('./val_image/val.txt')
if os.path.exists('./test_image/test.txt'):
	os.remove('./test_image/test.txt')

# get dirs and val_dirs and test_dirs
dirs, _ = get_dirinfo('./video_data')
testnum = int(len(dirs) * test_split)
valnum = int(len(dirs) * val_split)

test_dirs = random.sample(dirs, testnum)
for i in test_dirs:
	dirs.remove(i)

val_dirs = random.sample(dirs, valnum)
for i in val_dirs:
	dirs.remove(i)

print('get %d train data, %d val data, %d test data' % (len(dirs), len(val_dirs), len(test_dirs)))
# move image to goal dir
# train data
for i in dirs:
	_, files = get_dirinfo(i)
	for j in files:
		# for image j, find his label and rename it
		newname =  str(j[-1 * len('S001C001P003R002A058/064.jpg'):]).replace(r"/", '-')
		copy(j, './train_image/' + newname)
		label = int(str(j[-11:-8])) - 1
		with open('./train_image/train.txt', 'a+') as f:
			f.write(newname + ' ' + str(label) + '\n')

# val data
for i in val_dirs:
	_, files = get_dirinfo(i)
	for j in files:
		# for image j, find his label and rename it
		newname =  str(j[-1 * len('S001C001P003R002A058/064.jpg'):]).replace(r"/",'-')
		copy(j,'./val_image/'+ newname)
		label = int(str(j[-11:-8])) - 1
		with open('./val_image/val.txt', 'a+') as f:
			f.write(newname + ' ' + str(label) + '\n')

# test data
for i in test_dirs:
	_, files = get_dirinfo(i)
	for j in files:
		# for image j, find his label and rename it
		newname = str(j[-1 * len('S001C001P003R002A058/064.jpg'):]).replace(r"/", '-')
		copy(j, './test_image/' + newname)
		label = int(str(j[-11:-8])) - 1
		with open('./test_image/test.txt', 'a+') as f:
			f.write(newname + ' ' + str(label) + '\n')


