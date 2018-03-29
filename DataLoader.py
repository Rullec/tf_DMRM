#! /usr/bin/env python 
# -*- coding: utf-8 -*- 
# author: rullec
# date: 2018/3/28
# version:	v0.4 2018/3/28 move valdation data out of get_next_batch() func, it's more correct than before. 
#			v0.3 add assert to make sure the data format
#			v0.2 2018/3/20 -- use yield in dataloader

import os
import sys
import traceback
import random
import copy
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

reload(sys)
sys.setdefaultencoding('utf-8') 

class DataLoader(object):
	'''
	This class will read every subdir in datadir as an training example.
	an img LIST and an label LIST will be used to chareactrize an example, their index makes them one-to-one correspondence.

	static vars:
		subdirs: stored all of the subdirs' path in datadir
		nowpos: stored next data we should load, used for batches.
	'''
	def get_dirinfo(self, path):
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

	def __init__(self, datadir, test_split = 0.1):
		'''
		this function can get all subdir paths and stored them, and split valdation set on it randomly.
		parameters:
			datadir: path to datadir, such as "./video_dir"
		'''
		print('DataLoader start, reading data from %s.' % datadir)
		txtpath = './utils/subdirs.txt' # the index file that stored subdir paths
		imgpath = './utils/subfiles.txt' # the img index file
		val_txtpath = './utils/val_subdirs.txt' # the index file of valdation data subdirs
		#val_imgpath = './utils/val_subfiles.txt' # the index file of validation data images

		subdirs, subfiles = [], []
		valdirs, valfiles = [], []

		if os.path.exists(txtpath):
			print('subdirs.txt exists')
			with open(txtpath, 'r') as f:
				index = f.readlines()
				for i in range(0, len(index)):
					subdirs.append(index[i][0:-1])
		else:
			subdirs, _ = self.get_dirinfo(datadir)
			with open(txtpath, 'w') as f:
				for i in subdirs:
					f.write(i+'\n')
		
		# randomly choose the valdation subdirs and remove them from subfiles
		if os.path.exists(val_txtpath):
			print('val_subdirs.txt exists!')
			with open(val_txtpath, 'r') as f:
				index = f.readlines()
				for i in range(0, len(index)):
					valdirs.append(index[i][0:-1])
					subdirs.remove(index[i][0:-1])
					_, tmp = self.get_dirinfo(index[i][0:-1])
					[valfiles.append(j) for j in tmp]

		else: # else, we must choose some dirs from subdirs and remove them from subdirs
			const = int(len(subdirs) * test_split)
			valdirs = random.sample(subdirs, const)
			with open(val_txtpath, 'w') as f:
				for i in valdirs:
					subdirs.remove(i)
					f.write(i + '\n')
					# write these imgs into valfiles
					_, tmp = self.get_dirinfo(i)
					[valfiles.append(j) for j in tmp]

		if os.path.exists(imgpath):
			print('subfiles.txt exists.')
			with open(imgpath, 'r') as f:
				index = f.readlines()
				for i in range(0, len(index)):
					subfiles.append(index[i][0:-1])
		else:
			with open(imgpath, 'w') as f:
				for i in subdirs:
					_ , files = self.get_dirinfo(i)
					for j in files:
						subfiles.append(j)
						f.write(j + '\n')
		
		print('there are %d training samples, %d images, and %d testing samples, %d testing images in %s' % (len(subdirs), len(subfiles), len(valdirs), len(valfiles), datadir))
		self.subdirs = subdirs
		self.subfiles = subfiles
		self.valdirs = valdirs
		self.valfiles = valfiles

	def get_validation_data(self, NeedNum):
		'''
		get_validation_data, just given NeedNum
		this function will return val_img, val_label
		'''
		val_img, val_label = [], []
		try:
			val_files = random.sample(self.subfiles, NeedNum)
			# get validation sample in val_files
			for m in val_files:
				jpg = load_img(m)
				np_image = np.expand_dims(img_to_array(jpg), axis=0)
				val_img.append(np_image)
				onehot = np.zeros([1,60], dtype=int)
				onehot[0, (int)(m[-10:-8])-1] = 1
				val_label.append(onehot)
			self.loader_assert(val_label, val_img)
			print('get valdation data: %d images' % NeedNum)
		except Exception as e:
			traceback.print_exc()
		return val_img, val_label

	def get_test_data(self, NeedNum=0.01):
		NeedNum = int(len(self.valfiles)*NeedNum)
		test_img, test_label = [], []
		print(len(self.valfiles))
		try:
			test_files = random.sample(self.valfiles, NeedNum)
			# get validation sample in val_files
			for m in test_files:
				jpg = load_img(m)
				np_image = np.expand_dims(img_to_array(jpg), axis=0)
				test_img.append(np_image)
				onehot = np.zeros([1,60], dtype=int)
				onehot[0, (int)(m[-10:-8])-1] = 1
				test_label.append(onehot)
			self.loader_assert(test_label, test_img)
			print('get test data: %d images' % NeedNum)
		except Exception as e:
			traceback.print_exc()
		return test_img, test_label

	
	def get_next_batch(self, batch_size=32, datatype='single', simplify = 0.1, val_split = 0.1):
		'''
	this function aims at get next batch.
	we will return validation data defaultly.
	'''
		if 'single'!=datatype:
			print('warining: your datatype is %s, so val_label and val_img will return None.', datatype)
		for i in range(0, len(self.subdirs), batch_size):
			try:
				label, img, name = [], [], []
				for j in range(i, i + batch_size):
					if j==len(self.subdirs):
						print('Attention! Now that, you have read the dataset completely')
						yield img, label, name
						self.shuffle()
						break
		
					path = self.subdirs[j]
					_, files = self.get_dirinfo(path)
		
					# simplify - get key images' path in video
					new_files = []
					for i in xrange(0, int(round(simplify * len(files)))):
						const = int(round(1.0/simplify))
						if const * i >= len(files):
							new_files.append(files[-1])
						else:
							new_files.append(files[i * const])
					files = new_files
					
					# get the ith training sample label 
					onehot = np.zeros([1,60], dtype=int)
					onehot[0, (int)(path[-2:]) - 1 ] = 1
					if datatype!='single':
						label.append([np.array(onehot) for i in xrange(len(files))])
					else:
						[label.append(np.array(onehot)) for i in xrange(len(files))]

					# get the ith sample name
					name.append(path[-20:])

					# get the ith training sample images
					subdata = []
					for m in files:
						if '.jpg'==m[-4:]:
							try:
								jpg = load_img(m)
								np_image = np.expand_dims(img_to_array(jpg), axis=0)
								if datatype=='single':
									img.append(np_image)
									subdata.append([])
								else:
									subdata.append(np_image)
							except:
								traceback.print_exc()
								continue
					if 'single'!=datatype:
						img.append(subdata)

					# assert
					self.loader_assert(label, img, datatype)
					
					print('load %d sample succ, training samples num is %d' % (j, len(subdata)))
				yield img, label, name
			except Exception as e:
				traceback.print_exc()
				continue
				# How to process these exceptions will be disuceesed in future.

	def loader_assert(self, label, img, datatype='single'):

		if 0==len(label) or 0==len(img):
			raise ValueError('img and label is empty in loader_assert function!')
		subdata = img[-1]
		if 'single'!=datatype:
			# subdata must be a list and its length is equal to len(files)
			assert(isinstance(subdata,list))
			for i in subdata:
				assert(isinstance(i, np.ndarray))
				assert((1, 224, 224, 3)==i.shape)
			
			# label[-1] must be a list and its length is equal to len(files)
			# and every element in label[-1] must be a one-hot np array.
			for i in label[-1]:
				assert(isinstance(i, np.ndarray))
				assert((1,60)==i.shape)
		else:
			for i in label:
				assert(isinstance(i, np.ndarray))
				assert((1,60)==i.shape)
			for i in img:
				assert(isinstance(i, np.ndarray))
				assert((1, 224,224,3)==i.shape)
	
	def shuffle(self):
		'''
	this function can help you shuffle datasets, after any epoch.
	'''
		random.shuffle(self.subdirs)

if __name__ =='__main__':
	'DataLoader.py test start!!!'
	loader = DataLoader('./video_data')

	for img, label, name in loader.get_next_batch(batch_size=32, simplify=0.1):
		print(len(img), len(label))
	img, label = loader.get_validation_data(100)
	loader.shuffle()
