import csv
import cv2
import numpy as np
from scipy import ndimage

# read driving log from csv file
lines = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

# creat list of images and corresponding measurements from log
images = []
measurements = []
for line in lines:
    current_path = '/opt/carnd_p3/data/' + line[0]
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# augment images and measurements by flipping the iamge and negating the measurement
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1)

# convert features and labels into numpy array for training
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# creat model using keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)

# save the model
model.save('model.h5')
exit()