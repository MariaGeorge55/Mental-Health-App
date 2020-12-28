import os
import numpy as np
import cv2 
from keras.layers import Input, Conv2D, Activation, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential, Model
from keras.utils import to_categorical
import tensorflow.lite as lite
import tensorflow as tf



classes={0:'angry',1:'disgusted',2:'fearful',3:'happy',4:'neutral',5:'sad',6:'surprised'}

def construct_cnn():

    
    model = Sequential()
    model.add(Conv2D(12, (2,2),input_shape=(128, 128, 3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(32, (2,2),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(7,activation='softmax'))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    tflite_model = tf.keras.models.load_model('D:\CNNGrad\model_weights\model_weights.0006--1.5505--0.4683.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
    tflite_save = converter.convert()
    open("generated.tflite", "wb").write(tflite_save)
    return model
 

def load_existing_model_weights():
    model = construct_cnn()
    import os
    from pathlib import Path
    user_home = str(Path.home())
    model.load_weights("D:/CNNGrad/model_weights/model_weights.0006--1.5505--0.4683.h5")
    return model

size = 128,128
def load_image(file) :
    print('hi')
    try:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,size)
        return img
    except:
        return None
    

def classify_flower(file):
    model = load_existing_model_weights()

    img = load_image(file)

    img = np.asarray(img)
    img= img/255.0
    img=img.reshape(-1,128,128,3)
    estm= model.predict_proba(img,batch_size=32)
    Cvalue=model.predict_classes(img,batch_size=32)

    return Cvalue

cls = classify_flower('D:/CNNGrad/im1.png')

print(classes[cls[0]] )