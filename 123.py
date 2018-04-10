import numpy as np
import os 
with open('./utils/val_subdirs.txt', 'r') as f:
	test = f.readlines()
with open('./utils/subdirs.txt', 'r') as f:
	train = f.readlines()
path = './utils/cnn_features/'
test_file = []
train_file = []

if os.path.exists('./utils/cnn_features/train.txt'):
	os.remove('./utils/cnn_features/train.txt')
if os.path.exists('./utils/cnn_features/test.txt'):
	os.remove('./utils/cnn_features/test.txt')

for i in test:
	if i in train:
		train.remove(i)
		print('remove %s' % i )
	tmppath = path + i[-1*len('S001C002P003R002A035')-1:-1] + '.npy'
	test_file.append(tmppath)

with open('./utils/cnn_features/test.txt', 'w') as f:
	for i in test_file:
		f.write(i + '\n')
		print('test add %s' % i)

for i in train:
	tmppath = path + i[-1*len('S001C002P003R002A035')-1:-1] + '.npy'
	train_file.append(tmppath)

with open('./utils/cnn_features/train.txt', 'w') as f:
	for i in train_file:
		f.write(i + '\n')
		print('train add %s' % i)
