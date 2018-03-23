#! /usr/bin/env python 
# -*- coding: utf-8 -*- 
# author: rullec
# date: 2018/3/20
# version: v0.1

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

	def get_next_batch(self, batch_size=32):
		'''
	this function aims at get next batch.
	'''
		if self.nowpos == len(self.subdirs):
			print('Attention! Now that, you have read the dataset completely.if you want to read it again, please call obj.reset()!')
			return None
		left = self.nowpos
		right = self.nowpos + batch_size
		if right>len(self.subdirs):
			right = self.subdirs
		label, img, name = [], [], []
		for i in range(left, right):
			try:
				path = self.subdirs[i]
				_, files = self.get_dirinfo(path)
				#print(files)
				# get the ith example label
				label.append((int)(path[-2:]))
				name.append(path[-20:])
				# get the ith example images
				subdata = []
				for j in files:
					if '.jpg'==j[-4:]:
						jpg = load_img(j)
						np_image = img_to_array(jpg)
						subdata.append(np_image)
						#print(np_image.shape)
				img.append(subdata)
				print('load %d sample succ, jpg num is %d.' % (i, len(subdata)))
			except Exception as e:
				traceback.print_exc()
				continue
			# How to process these exceptions will be disuceesed in future.
		self.nowpos = right
		return img, label, name

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
	img, label, name = loader.get_next_batch()
	loader.shuffle()
