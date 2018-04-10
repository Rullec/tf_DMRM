import os
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking
from keras.callbacks import ModelCheckpoint

random.seed(0)
np.random.seed(7)

# if this variable is True, we will choost val test data again from the dataset.
# if it's true, we will read record from files
READ_FROM_FILE = True

# index file path
data_dir = './utils/cnn_features/'
test_file = data_dir + 'test.txt'
val_file = data_dir + 'val.txt'
train_file = data_dir + 'train.txt'

# list stored file name 
test_files, train_files, val_files = [], [], []

# stored data 
train_data, train_label = [], []
test_data, test_label = [], []
val_data, val_label = [], []

def get_dirinfo(path):
	filelist = []
	dirlist = []
	# if this path existss and it is a dir(not a file)
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
		raise('the directory ' + path + ' is not exists!')
	filelist.sort()
	dirlist.sort()
	return dirlist, filelist

def load_file(filepath):
	tmp = np.expand_dims(np.load(filepath), axis = 0)
	print('load file output dims: %s ' % str(tmp.shape))
	label = np.zeros([1,60], dtype=int)
	label[0][int(filepath[-7:-4])-1] = 1
	return tmp, label

def load_data(train_files, val_files, test_files):
	for i in test_files:
		a, b = load_file(i.strip())
		test_data.append(a)
		test_label.append(b)
	for i in val_files:
		a, b = load_file(i.strip())
		val_data.append(a)
		val_label.append(b)
	for i in train_files:
		print('load train file: ' + i)
		a, b = load_file(i.strip())
		train_data.append(a)
		train_label.append(b)
	print('load data succ! get %d traindata, %d valdata, %d testdata.' % (len(train_data), len(val_data), len(test_data)))

# Completion_input
def completion_input(data, label, max_review_length = 50):
	X = data
	Y = label
	for i in xrange(0, len(data)):
		print(X[i].shape)
		X[i] = np.lib.pad(X[i], ((0,0),(max_review_length - X[i].shape[1], 0),(0,0)), 'constant', constant_values=(0, 0))
		print(X[i].shape)

	X = np.concatenate(X, axis=0)#np.random.rand(60, 10, 4096)
	Y = np.concatenate(Y, axis=0)#np.random.rand(60, 60)
	return X, Y
###### below is main stream #######

with open('train.txt', 'r') as f:
	a = f.readlines()
	for i in a:
		train_files.append(i)
	print('train num %d' % len(train_files))

with open(test_file, 'r') as f:
	test_files = f.readlines()

with open(val_file, 'r') as f:
	val_files = f.readlines()

print('read val file succ, %d sample' % len(val_files))
print('read test file succ, %d sample' % len(test_files))
print('read train file succ, %d sample' % len(train_files))
load_data(train_files, val_files, test_files)

# max frame length for one video
max_review_length = 50

#LSTM structure
model = Sequential()
#model.add(Masking(mask_value=np.zeros([1,4096]), input_shape=(max_review_length, 4096)))
model.add(LSTM(100, input_shape=(max_review_length, 4096), dropout_W=0.3, dropout_U=0.3))
model.add(Dense(60, activation='softmax', name='softmax-60'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#preprocess input data
train_data, train_label = completion_input(train_data, train_label, max_review_length)
test_data, test_label = completion_input(test_data, test_label, max_review_length)
val_data, val_label = completion_input(val_data, val_label, max_review_length)

#set checkpoint
checkpoint = ModelCheckpoint('./utils/checkpoints/lstm/weights.epoch{epoch:03d}-val_acc{val_acc:.5f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='max')

#train model
model.fit(train_data, train_label, validation_data = (val_data, val_label), epochs=200, callbacks = [checkpoint], verbose=1, batch_size = 32)

# evaluate
scores = model.evaluate(test_data, test_data, batch_size = 32)
print(scores)

