import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D

DATA_DIR = 'data6/'
IMAGE_PATH = DATA_DIR + 'IMG/'
IMAGE_SEP = '\\'
STEERING_CORRECTION = 0.25

def load_data(filename="driving_log.csv"):
    print("loading csv data...")
    lines = []
    with open(DATA_DIR + filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines

def generator(samples, batch_size=32):
    """
    Generate batch samples and yield on the fly. Load center, left and right
    images along with their corresponding angle measurements. Augment samples
    by flipping each image and inverting their angle values.
    """
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                file_center = batch_sample[0].split(IMAGE_SEP)[-1]
                file_left = batch_sample[1].split(IMAGE_SEP)[-1]
                file_right = batch_sample[2].split(IMAGE_SEP)[-1]

                img_center = cv2.imread(IMAGE_PATH + file_center)
                aug_img_center = cv2.flip(img_center, 1)

                img_left = cv2.imread(IMAGE_PATH + file_left)
                aug_img_left = cv2.flip(img_left, 1)

                img_right = cv2.imread(IMAGE_PATH + file_right)
                aug_img_right = cv2.flip(img_right, 1)

                steering_center = float(batch_sample[3])
                steering_left = steering_center + STEERING_CORRECTION
                steering_right = steering_center - STEERING_CORRECTION

                images.extend([img_center, img_left, img_right])
                images.extend([aug_img_center, aug_img_left, aug_img_right])

                angles.extend([steering_center, steering_left, steering_right])
                angles.extend([-steering_center, -steering_left, -steering_right])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def nvidia_architecture():
    """
    Nvidia architecture implementation using Keras. The model consists of 4
    convolutions and 5 fully connected layers.
    Preprocessing consists of image pixel normalization and cropping.
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,5,5, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

def train(model, epochs=5):
    print("training...")
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),\
            validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=epochs)

    print("save")
    model.save('model.h5')

if __name__ == '__main__':
    samples = load_data()
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = nvidia_architecture()
    train(model, epochs=5)
