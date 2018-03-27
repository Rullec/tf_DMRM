# date: 2018/3/23
# version: v0.2 I add the 64 lines after pop layers i add a Model function. it's solve the problem that lavk of node config when saving the model at last
'''
the ImageNet's weight is not suitable with our dataset, we must train it again to adjust its weights.
Our dataset have 60 classes, so we can change its output-Layers to (None, 60)
use SGD or Adams method to train it, Remember: You must plot its loss function. 
at last, you must save your network weights and Layers.Otherwise you will get nothing.
'''
from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model, load_model, save_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import numpy as np
from DataLoader import DataLoader

def pop_model(model):
	if not model.outputs:
		raise Exception('Sequential model cannot be poped: model is empty')
	
	model.layers.pop()
	if not model.layers:
		model.outputs = []
		model.inbound_nodes = []
		model.outbound_nodes = []
	else:
		model.layers[-1].outbound_nodes = []
		model.outputs = [model.layers[-1].output]
	#model.built = False

def splitlist(list):

	alist = []
	a = 0
	
	for sublist in list:
		try:
			for i in sublist:
				alist.append(i)
		except:
			alist.append(sublist)
	for i in alist:
		if type(i) == type([]):
			a =+ 1
			break
	if 1==a:
		return splitlist(alist)
	if 0==a:
		return alist

# Cut the model's cnn and pooling layers, get rid of all FC layers by include_top = False
# This model have no pop method 
base_model = vgg16.VGG16(weights = 'imagenet')
pop_model(base_model)
pop_model(base_model)
pop_model(base_model)
pop_model(base_model)

'''
# Define Input image format as 'tf': 224*224*3
inputs = Input(shape = (224,224,3), name = 'image_input')
'''
my_model = Model(inputs=base_model.inputs, outputs=base_model.outputs)

'''
# Use the generated model
tmp = base_model(inputs)
print(base_model.outputs)
#base_model.trainable = False
'''

# Add the fully-connected layers
tmp = Flatten(name='flatten')(my_model.output)
tmp = Dense(4096, activation='relu', name='fc1')(tmp)
tmp = Dense(4096, activation='relu', name='fc2')(tmp)
outputs = Dense(60, activation='softmax', name='predictions')(tmp)

my_model = Model(inputs=my_model.inputs, outputs=outputs)

# Create and Compile your own model 
#my_model = Model(inputs, outputs)
sgd = SGD(lr = 0.001, momentum = 0.9, decay = 0.0)
my_model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
my_model.summary()

# Create checkpoints
checkpoint = ModelCheckpoint('./utils/checkpoints/weights.epoch{epoch:02d}-val_acc{val_acc:.5f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Begin to train and load data
loader = DataLoader('./video_data')

batch_size = 256
#my_model = load_model('utils/checkpoints/weights.01-0.00.hdf5')
for data, label, val_data, val_label, name in loader.get_next_batch(batch_size, datatype='single', val_split=0.1, simplify=0.1):
	#for i in data:
	#	print(len(i))
	#my_model.fit(x=np.expand_dims(data[0], axis=0) ,y=label[0], batch_size=32, epochs=10, validation_split=0.05, shuffle=True)
	#print('old###########data')
	#print(type(data), len(data),  data[0].shape)
	#print('old###########label')
	#print(type(label), len(label),  label[0].shape)
	label = np.concatenate(label, axis=0)
	data = np.concatenate(data, axis=0)
	#print('new###########data')
	val_label = np.concatenate(val_label, axis=0)
	val_data = np.concatenate(val_data, axis=0)
	#print(type(data), len(data),  data[0].shape)
	#print('new###########label')
	#print(type(label), len(label),  label[0].shape)
	my_model.fit(x=data, y=label, batch_size=32, epochs=20, callbacks=[checkpoint], validation_data=(val_data, val_label), shuffle=True)
