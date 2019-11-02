import cv2
import matplotlib.pyplot as plt
import os
from keras.applications import DenseNet121
import numpy as np
from keras.models import *
from keras.layers import Dense, Dropout, Input
from keras.utils import np_utils
from keras.optimizers import *
import random
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
random.seed(1)


pm = "D:\Python Scripts\PALM\data\PM"  
X_train=[]
y_train = []

traingen=ImageDataGenerator(rescale=1./255,
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
                    
valgen=ImageDataGenerator(rescale=1./255,
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

seed  =1 


## Loading Dataset
train_set= traingen.flow_from_directory(directory='PALM/t1',target_size=(224,224),
                                            batch_size=2,class_mode='categorical',seed=seed)
val_set = valgen.flow_from_directory(directory='PALM/v1',target_size=(224,224),
                                            batch_size=2,class_mode='categorical',seed=seed)


## MODEL Definition
def def_model(pretrained_weights =None):
    input=Input(shape=(224,224,3))
    res=DenseNet121(input_tensor=input,include_top=False,pooling='max',weights='imagenet')
    x=Dense(1024,activation='relu')(res.output)
    x=Dropout(rate=0.4)(x)
    out=Dense(2,activation='softmax')(x)
    model=Model(inputs = res.input,outputs = out)
    model.compile(optimizer=Adam(lr=1e-5),loss="categorical_crossentropy",metrics=['accuracy'])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model



## Training Model
model = def_model(pretrained_weights = 'PALM/DenseNet/Weights_DenseNet121/Densenet121.01-0.4548-0.8269.hdf5')#'PALM/DenseNet/Weights_DenseNet121/Densenet121.01-0.3693-0.9038.hdf5'
checkpoint =model_checkpoint = ModelCheckpoint('PALM/DenseNet/Weights_DenseNet121/Densenet121.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',mode='max',factor=0.3, patience=10, min_lr=1e-9, verbose=1)
callback_list=  [checkpoint,reduce_lr]
history=model.fit_generator(generator = train_set,steps_per_epoch=171,epochs=50,callbacks=callback_list,
                                    validation_data=val_set,validation_steps= 26,verbose=1)
