#! /usr/bin/env python 
# -*- coding: utf-8 -*- 
# author: rullec
# date: 2018/3/20
# version: v0.1
import sys 
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8') 

image_raw_data = tf.gfile.FastGFile('/home/penglu/Desktop/11.jpg').read()  
image = tf.image.decode_jpeg(image_raw_data) #图片解码  
  
print image.eval(session=tf.Session())  
