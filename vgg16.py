import traceback
import keras
from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
import DataLoader

#Load the VGG model and take out the layers you need.
base_model = vgg16.VGG16(weights='imagenet', include_top=False)
vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# load all data from datadir
dataloader = DataLoader.DataLoader('./video_data')
# img is a list = [video1, video2, video3], and every element in img such as video1 = [video1, frame2, ... , frameN]
# every video is a numpy.ndarray element, frame.shape = (224, 224, 3)

for data, label, name in dataloader.get_next_batch():
	try:	
		for i, video in enumerate(data):
			video = np.asarray(video)
			x = preprocess_input(video, mode='tf')
			# output
			feature = vgg_model.predict(x)
			path = './utils/cnn_features/'+str(name[i])+'.npy'
			np.save(path, feature)
			print(path+' saved succ!')
		if data is None:
			print('cnn have exetracted all features, done.')
			break
	except Exception as e:
		traceback.print_exc()
		pass
