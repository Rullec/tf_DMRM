# -*- coding: utf-8
# author: rullec
# date: 2018/3/21
# version: v0.1

'''
this script is jpg_squeezer.py
For example, it can help you squeeze an 2096*1080 .jpg image to 224*224 image.
'''

import Image as image
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def clipResizeImg(**args):
	'''
this function help you clip and resizing image file.

@function author: fc_lamp
@author blog: http://fc-lamp.blog.163.com/

parameters:
	ori_img: the source img path
	dst_img: the destine img path
	dst_w : the width of target img
	dst_h: the height of target img
	'''

	args_key = {'ori_img':'','dst_img':'','dst_w':'','dst_h':'','save_q':75}
	arg = {}
	for key in args_key:
		if key in args:
			arg[key] = args[key]
	im = image.open(arg['ori_img'])
	ori_w,ori_h = im.size
	dst_scale = float(arg['dst_h']) / arg['dst_w'] #目标高宽比
	ori_scale = float(ori_h) / ori_w #原高宽比
	if ori_scale >= dst_scale:
		#过高
		width = ori_w
		height = int(width*dst_scale)
		x = 0
		y = (ori_h - height) / 3
	else:
		#过宽
		height = ori_h
		width = int(height*dst_scale)
		x = (ori_w - width) / 2
		y = 0
	#裁剪
	box = (x,y,width+x,height+y)
	#这里的参数可以这么认为：从某图的(x,y)坐标开始截，截到(width+x,height+y)坐标
	#所包围的图像，crop方法与php中的imagecopy方法大为不一样
	newIm = im.crop(box)
	im = None
	#压缩
	ratio = float(arg['dst_w']) / width
	newWidth = int(width * ratio)
	newHeight = int(height * ratio)
	newIm.resize((newWidth,newHeight),image.ANTIALIAS).save(arg['dst_img'],quality=arg['save_q'])

def main():
	num = 0
	for i in sys.argv[1:] :
		# source image
		ori_img = i
		# destine image
		dst_img = i
		# shape of destine image 
		dst_w = 224
		dst_h = 224 
		# image quality
		save_q = 95

		clipResizeImg(ori_img=ori_img, dst_img=dst_img, dst_w=dst_w, dst_h=dst_h, save_q=save_q)
		num = num + 1
		print('process %s succ! process for %f percents' % (dst_img, num*100.0/len(sys.argv[1:])))
	return

main()
