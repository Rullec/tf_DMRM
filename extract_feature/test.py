from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model, load_model, save_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import numpy as np
from DataLoader import DataLoader
import sys
import matplotlib.pyplot as plt

loader = DataLoader('video_data')
data,label = loader.get_test_data(0.05)
label = np.concatenate(label, axis=0)
data = np.concatenate(data, axis=0)

sumx = []
sumy = []
for index, i in enumerate(sys.argv[1:]):
	print('testing on ' + str(i) + '...')
	my_model = load_model('../utils/checkpoints/' + i)
	res = my_model.evaluate(x = data, y = label, batch_size=32)
	with open('../utils/validation.txt', 'a') as f:
		tmp = str(i) +' loss=' + str(res[0]) + ', val_acc=' + str(res[1]) + '\n'
		print(tmp)
		f.write(tmp)
	sumx.append(index)
	sumy.append(res[1])

#plt.plot(sumx, sumy)
#plt.show()
