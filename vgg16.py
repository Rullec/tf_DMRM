import traceback
import keras
from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model, load_model
import numpy as np
import DataLoader

#Load the VGG model and take out the layers you need.
#base_model = vgg16.VGG16(weights='imagenet', include_top=False)
base_model = load_model('utils/checkpoints/weights.epoch01-val_acc0.97874.hdf5')
vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# load all data from datadir
dataloader = DataLoader.DataLoader('./video_data')
# img is a list = [video1, video2, video3], and every element in img such as video1 = [video1, frame2, ... , frameN]
# every video is a numpy.ndarray element, frame.shape = (224, 224, 3)

for data, label, name in dataloader.get_next_batch(batch_size=16, simplify=1, datatype='notsingle'):
	try:	
		for i, video in enumerate(data):
			video = np.concatenate(video, axis=0)
			# output
			feature = vgg_model.predict(video)
			#print(feature.shape)
			path = './utils/cnn_features/'+str(name[i])+'.npy'
			np.save(path, feature)
			print(path+' saved succ!')
		if data is None:
			print('cnn have exetracted all features, done.')
			break
	except Exception as e:
		traceback.print_exc()
		pass
