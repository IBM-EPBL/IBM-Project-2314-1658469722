

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,rotation_range=180,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
x_train=train_datagen.flow_from_directory(r'E:\journal\Nalaya thiran\Dataset\Dataset\train_set', target_size=(128,128),batch_size=32,
                                       class_mode='binary')

x_test=train_datagen.flow_from_directory(r'E:\journal\Nalaya thiran\Dataset\Dataset\test_set', target_size=(128,128),
                                       batch_size=32,
                                       class_mode='binary')
x_train.class_indices

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D,MaxPooling2D, Flatten
import warnings
warnings.filterwarnings('ignore')


model=Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(128,128,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.summary()
model.add(Dense(150,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuacy'])
len(x_train)
len(x_test)
model.fit_generator(x_train,steps_per_epoch=len(x_train),epochs=10,
                  validation_data=x_test,validation_steps=len(x_test))

import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
model.save('forestfire.h5')
model=load_model('forestfire.h5')
testImg = image.load_img(r'E:\journal\Nalaya thiran\Dataset\Dataset\test_set\with fire\louisiana_forest_fire.jpg', target_size = (128,128))
testImg
arrayImg = image.img_to_array(testImg)
arrayImg
x = np.expand_dims(arrayImg , axis = 0)
x




images = np.vstack([x])
pred=model.predict(images)
pred
x_train.class_indices
if (pred[0] > 0.5):
    account_sid='AC2e95b7df475c1def9cd2dd30d2887abc'
    auth_token='5f20df1bea0e666d6dfad7ce8e0bf631'
    client=Client(account_sid,auth_token)
    message=client.messages.create(body='Forest Fire is detected,stay alert',
    from_='+13465155048',
    to='+91 9360069501')
    print("fire detected")
    print("SMS sent!")
else:
    print("No danger")

from twilio.rest import Client
import cv2
video=cv2.VideoCapture(r'E:\journal\Nalaya thiran\video.mp4')
name=['forest','with fire']
account_sid='AC2e95b7df475c1def9cd2dd30d2887abc'
auth_token='5f20df1bea0e666d6dfad7ce8e0bf631'
client=Client(account_sid,auth_token)
message=client.messages.create(body='Forest Fire is detected,stay alert',
from_='+13465155048',
to='+91 9360069501')
print(message.sid)
get_ipython().system('pip install opencv-python')
get_ipython().system('pip install opencv-contrib-python --user')


import keras
from tensorflow.keras.utils import load_img, img_to_array
while(1):
    success,frame=video.read()
    img=keras.utils.load_img("image.jpg")
    img=cv2.resize(frame,(128,128))
    x=keras.utils.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    dim=(128,128)
    pred = model.predict(x)
    p=pred[0]
    print(pred)
if pred[0]==0:
    account_sid='AC2e95b7df475c1def9cd2dd30d2887abc'
    auth_token='5f20df1bea0e666d6dfad7ce8e0bf631'
    client=Client(account_sid,auth_token)
    message=client.messages.create(body='Forest Fire is detected,stay alert',
    from_='+13465155048',
    to='+91 9360069501')
    print("fire detected")
    print("SMS sent!")
else:
    print("No Danger")
video.release()
cv2.destroyAllWindows()






