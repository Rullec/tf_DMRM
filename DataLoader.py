#! /usr/bin/env python 
# -*- coding: utf-8 -*- 
# author: rullec
# date: 2018/3/20
# version: v0.3 add assert to make sure the data format	/ v0.2 -- use yield in dataloader

import os
import sys
import traceback
import random
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
		filelist.sort()
		dirlist.sort()
		return dirlist, filelist

	def __init__(self, datadir):
		'''
		this function can get all subdir paths and stored them.
		parameters:
			datadir: path to datadir, such as "./video_dir"
		'''
		print('DataLoader start, reading data from %s.' % datadir)
		txtpath = './utils/subdirs.txt' # the index file that stored subdir paths
		subdirs = []
		if os.path.exists(txtpath):
			with open(txtpath, 'r') as f:
				index = f.readlines()
				for i in range(0, len(index)):
					subdirs.append(index[i][0:-1])
		else:
			subdirs, _ = self.get_dirinfo(datadir)
			with open('./utils/subdirs.txt', 'w') as f:
				for i in subdirs:
					f.write(i+'\n')
		print('there are %d examples in %s.' % (len(subdirs), datadir))
		self.subdirs = subdirs
		self.nowpos = 0

	def get_next_batch(self, batch_size=2, datatype='single', simplify = 0.1):
		'''
	this function aims at get next batch.
	'''
			
		'''		
		if self.nowpos == len(self.subdirs):
			return None
		left = self.nowpos
		right = self.nowpos + batch_size
		if right>len(self.subdirs):
			right = self.subdirs
		'''
		for i in range(0, len(self.subdirs), batch_size):
			try:
				label, img, name = [], [], []
				for j in range(i, i + batch_size):
					if j==len(self.subdirs):
						print('Attention! Now that, you have read the dataset completely.if you want to read it again, the obj will call obj.reset() automatically!')
						yield img, label, name
						self.shuffle()
		
					path = self.subdirs[j]
					_, files = self.get_dirinfo(path)

					# simplify - get some key image in video
					new_files = []
					for i in xrange(0, int(round(simplify * len(files)))):
						const = int(round(1.0/simplify))
						if const * i >= len(files):
							new_files.append(files[-1])
						else:
							new_files.append(files[i * const])
					files = new_files

					# get the ith example label
					onehot = np.zeros([1,60],dtype=int)
					onehot[0, (int)(path[-2:])-1] = 1
					if datatype!='single':
						label.append([np.array(onehot) for i in xrange(len(files))])
					else:
						[label.append(np.array(onehot)) for i in xrange(len(files))]
					# get the ith example name
					name.append(path[-20:])

					# get the ith example images
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

					print('load %d sample succ, jpg num is %d.' % (j, len(subdata)))
					self.nowpos = self.nowpos + 1
				yield img, label, name
			except Exception as e:
				print(e)
				traceback.print_exc()
				continue
				# How to process these exceptions will be disuceesed in future.

	def loader_assert(self, label, img, datatype='single'):

		subdata = img[-1]
		if 'single'!=datatype:
			# subdata must be a list and its length is equal to len(files)
			assert(isinstance(subdata,list))
			assert(len(subdata)==len(files))
			for i in subdata:
				assert(isinstance(i, np.ndarray))
				assert((1, 224, 224, 3)==i.shape)
			
			# label[-1] must be a list and its length is equal to len(files)
			# and evert element in label[-1] must be a one-hot np array.
			assert(len(files)==len(label[-1]))
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
		self.reset()
		random.shuffle(self.subdirs)

	def reset(self):
		'''
	this function can 'reset' the dataset
		just forget nowpos in datasets.
	'''
		self.nowpos = 0
	
if __name__ =='__main__':
	'DataLoader.py test start!!!'
	loader = DataLoader('./video_data')

	for img, label, name in loader.get_next_batch(batch_size=32):
		pass#print(name)
	loader.shuffle()
