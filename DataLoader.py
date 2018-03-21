#! /usr/bin/env python 
# -*- coding: utf-8 -*- 
# author: rullec
# date: 2018/3/20
# version: v0.1
import sys
import traceback
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8') 

# 
filenames=['images/000001.jpg','images/000002.jpg','images/000003.jpg','images/000004.jpg']
labels=[1,0,1,0]

filename_queue=tf.train.string_input_producer(filenames)

reader=tf.WholeFileReader()
filename, content = reader.read(filename_queue)
images=tf.image.decode_jpeg(content, channels=3)
images=tf.cast(images, tf.float32)
resized_images=tf.image.resize_images(images, 224, 224)

image_batch, label_batch=tf.train.batch([resized_images, labels], batch_size=2)
print image.eval(session=tf.Session())  

class DataLoader(object):
	'''
	This class will read every subdir in datadir as an training example.
	an img LIST and an label LIST will be used to chareactrize an example, their index makes them one-to-one correspondence.
	'''
	def __init__(self, datadir)
		try: 
			print('DataLoader start...read data from %s.' % datadir)
			subdirs, files = get_dirinfo(datadir)
			print('there are %d examples in %s.' % (len(subdirs), datadir))
			for i in range(len(subdirs)):
				
		except Exception as e:
			traceback.print_exec()
			# How to process these exceptions will be disuceesed in future.

	def get_dirinfo(self, datadir)
	
