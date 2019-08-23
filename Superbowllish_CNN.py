from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf
import os, cv2, re, random
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import layers, models, optimizers

train_data_dir = 'superbowllsh/train/'
test_data_dir = 'superbowllsh/validation/test/'
train_images = [train_data_dir+i for i in os.listdir(train_data_dir)] 
test_images = [test_data_dir+i for i in os.listdir(test_data_dir)]

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

train_images.sort(key=natural_keys)
test_images.sort(key=natural_keys)

# dimensions of our images.
img_width = 341
img_height = 256
def prepare_data(list_of_images):
    x = [] # images as arrays
    y = [] # labels

    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image), (img_width, img_height), interpolation=cv2.INTER_CUBIC))
    for i in list_of_images:
        if 'dirty' in i:
            y.append(1)
        elif 'cleaned' in i:
            y.append(0)
            
    return x, y


X, Y = prepare_data(train_images)
print(K.image_data_format())

print(train_images)

print(X)
print(Y)

X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=1)

nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)



epochs = 50
batch_size = 16

input_shape = (img_height, img_width, 3)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False)
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(np.array(X_train), (Y_train), batch_size=batch_size)
validation_generator = val_datagen.flow(np.array(X_val), (Y_val), batch_size=batch_size)

print(np.array(X_train).shape)
X_train = np.array(X_train)


history = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples // 16,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=30 // 16)
model.save('first_model.h5')
model.save_weights('first_weights.h5')

X_test, Y_test = prepare_data(test_images)

test_generator = val_datagen.flow(np.array(X_test), batch_size=22)
prediction_probabilities = model.predict_generator(test_generator, steps = 30 ,verbose=1)
print(len(prediction_probabilities))
counter = range(0, 660)
solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})
cols = ['label']

for col in cols:
    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)

solution.to_csv("sample_submission.csv", index = False)
