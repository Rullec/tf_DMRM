from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense, BatchNormalization, Concatenate, add, MaxPooling2D, Dropout, Conv2D
from keras.models import Model, load_model, save_model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
import numpy as np
def model(weights=None):
    # model1-RGB spatial
    # model1-model1-block 1
    input1 = Input(shape=(224, 224, 3))
    model1x = Conv2D(64, (3, 3), activation='relu', padding='same', name='model1-block1_conv1')(input1)
    model1x = Conv2D(64, (3, 3), activation='relu', padding='same', name='model1-block1_conv2')(model1x)
    model1x = MaxPooling2D((2, 2), strides=(2, 2), name='model1-model1-block1_pool')(model1x)
    model1x = BatchNormalization()(model1x)
    
    # model1-model1-block 2
    model1x = Conv2D(128, (3, 3), activation='relu', padding='same', name='model1-block2_conv1')(model1x)
    model1x = Conv2D(128, (3, 3), activation='relu', padding='same', name='model1-block2_conv2')(model1x)
    model1x = MaxPooling2D((2, 2), strides=(2, 2), name='model1-block2_pool')(model1x)
    model1x = BatchNormalization()(model1x)
    
    # model1-block 3
    model1x = Conv2D(256, (3, 3), activation='relu', padding='same', name='model1-block3_conv1')(model1x)
    model1x = Conv2D(256, (3, 3), activation='relu', padding='same', name='model1-block3_conv2')(model1x)
    model1x = Conv2D(256, (3, 3), activation='relu', padding='same', name='model1-block3_conv3')(model1x)
    model1x = MaxPooling2D((2, 2), strides=(2, 2), name='model1-block3_pool')(model1x)
    model1x = BatchNormalization()(model1x)
    
    # model1-block 4
    model1x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model1-block4_conv1')(model1x)
    model1x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model1-block4_conv2')(model1x)
    model1x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model1-block4_conv3')(model1x)
    model1x = MaxPooling2D((2, 2), strides=(2, 2), name='model1-block4_pool')(model1x)
    model1x = BatchNormalization()(model1x)
    
    # model1-block 5
    model1x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model1-block5_conv1')(model1x)
    model1x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model1-block5_conv2')(model1x)
    model1x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model1-block5_conv3')(model1x)
    model1x = BatchNormalization()(model1x)
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # model2-optical flow temporal
    input2 = Input(shape=(224, 224, 10))
    # model2-model2-block 1
    model2x = Conv2D(64, (3, 3), activation='relu', padding='same', name='model2-block1_conv1')(input2)
    model2x = Conv2D(64, (3, 3), activation='relu', padding='same', name='model2-block1_conv2')(model2x)
    model2x = MaxPooling2D((2, 2), strides=(2, 2), name='model2-model2-block1_pool')(model2x)
    model2x = BatchNormalization()(model2x)
    
    # model2-model2-block 2
    model2x = Conv2D(128, (3, 3), activation='relu', padding='same', name='model2-block2_conv1')(model2x)
    model2x = Conv2D(128, (3, 3), activation='relu', padding='same', name='model2-block2_conv2')(model2x)
    model2x = MaxPooling2D((2, 2), strides=(2, 2), name='model2-block2_pool')(model2x)
    model2x = BatchNormalization()(model2x)
    
    # model2-block 3
    model2x = Conv2D(256, (3, 3), activation='relu', padding='same', name='model2-block3_conv1')(model2x)
    model2x = Conv2D(256, (3, 3), activation='relu', padding='same', name='model2-block3_conv2')(model2x)
    model2x = Conv2D(256, (3, 3), activation='relu', padding='same', name='model2-block3_conv3')(model2x)
    model2x = MaxPooling2D((2, 2), strides=(2, 2), name='model2-block3_pool')(model2x)
    model2x = BatchNormalization()(model2x)
    
    # model2-block 4
    model2x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model2-block4_conv1')(model2x)
    model2x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model2-block4_conv2')(model2x)
    model2x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model2-block4_conv3')(model2x)
    model2x = MaxPooling2D((2, 2), strides=(2, 2), name='model2-block4_pool')(model2x)
    model2x = BatchNormalization()(model2x)
    
    # model2-block 5
    model2x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model2-block5_conv1')(model2x)
    model2x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model2-block5_conv2')(model2x)
    model2x = Conv2D(512, (3, 3), activation='relu', padding='same', name='model2-block5_conv3')(model2x)
    model2x = BatchNormalization()(model2x)
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Fusion
    model1x = add([model1x, model2x])

    # model1 branch
    model1x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='model1-block5_pool')(model1x)
    model1x = Flatten(name='model1-flatten')(model1x)
    model1x = Dense(4096, activation='relu', name='model1-fc1')(model1x)
    model1x = Dropout(0.2)(model1x)
    model1x = Dense(4096, activation='relu', name='model1-fc2')(model1x)
    model1_output= Dropout(0.2)(model1x)
    
    # model2 branch
    model2x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='model2-block5_pool')(model2x)
    model2x = Flatten(name='model2-flatten')(model2x)
    model2x = Dense(4096, activation='relu', name='model2-fc1')(model2x)
    model2x = Dropout(0.2)(model2x)
    model2x = Dense(4096, activation='relu', name='model2-fc2')(model2x)
    model2_output = Dropout(0.2)(model2x)
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Fusion
    sum_output = add([model1_output, model2_output])
    outputs = Dense(60, activation='softmax', name='predictions')(sum_output)
    
    my_model = Model(inputs=[input1, input2], outputs = outputs)
    if weights:
        my_model.load_weights(weights)
        
    return my_model
    
my_model = model()

adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
my_model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
my_model.summary()
plot_model(my_model, to_file="model.png", show_shapes = False)
