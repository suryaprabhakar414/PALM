import cv2
import matplotlib.pyplot as plt
import os
from keras.applications import ResNet50
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.utils import np_utils
from keras.optimizers import Adam
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
random.seed(1)
print(os.getcwd())

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
## Loading Images
train_set= traingen.flow_from_directory(directory='PALM/train',target_size=(224,224),
                                            batch_size=2,class_mode='categorical',seed=seed)
val_set = valgen.flow_from_directory(directory='PALM/Val',target_size=(224,224),
                                            batch_size=2,class_mode='categorical',seed=seed)

## Model Definition
def def_model(pretrained_weights =None):
    input=Input(shape=(224,224,3))
    res=ResNet50(input_tensor=input,include_top=False,pooling='max',weights='imagenet')
    x=Dense(1024,activation='relu')(res.output)
    x=Dropout(rate=0.4)(x)
    out=Dense(2,activation='softmax')(x)
    model=Model(inputs = res.input,outputs = out)
    model.compile(optimizer=Adam(lr=1e-5),loss="categorical_crossentropy",metrics=['accuracy'])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

# Training
model = def_model(pretrained_weights = 'PALM\Resnet50\Weights_Resnet50\Resnet50.01-0.0901-0.9615.hdf5') 
checkpoint =model_checkpoint = ModelCheckpoint('PALM\Resnet50\Weights_Resnet50\Resnet50.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
callback_list=  [checkpoint]
history=model.fit_generator(generator = train_set,steps_per_epoch=171,epochs=50,callbacks=callback_list,
                                    validation_data=val_set,validation_steps= 26,verbose=1)
                                    
