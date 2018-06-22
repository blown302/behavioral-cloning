import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Conv2D, Cropping2D, Lambda
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def get_path(basePath, path):
    filename = path.split('/')[-1]
    return '{}/IMG/{}'.format(basePath, filename)


def pre_process_image(img):
    """
    Pre-processes an image array. Adds brightness and converts to RGB.
    :param img:
    :return:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 2] += 10
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            basePath = 'data/training/drive'
            images = []
            angles = []
            for line in batch_samples:
                measurement = float(line[3])

                # correction value for right and left cameras.
                correction = .07

                # factor to increase steering angle.
                measurement_boost_factor = .05

                # calculate steering angle.
                measurement = measurement + measurement * measurement_boost_factor

                # opens and pre-processes center image.
                center_path = get_path(basePath, line[0])
                center_image = cv2.imread(center_path)
                center_image = pre_process_image(center_image)

                # Adds center image and angle to batch.
                images.append(center_image)
                angles.append(measurement)

                # copies can flips center image. flipped images with augmented angle to batch.
                images.append(cv2.flip(center_image.copy(), 1))
                angles.append(measurement * -1)

                # opens and pre-processes left image.
                left_path = get_path(basePath, line[1])
                left_image = cv2.imread(left_path)
                left_image = pre_process_image(left_image)

                # augment angle for left by adding off-center correction.
                left_measurement = measurement + correction

                # add left image to batch.
                images.append(left_image)
                angles.append(left_measurement)

                # copies can flips left image. flipped images with augmented angle to batch.
                images.append(cv2.flip(left_image.copy(), 1))
                angles.append(left_measurement * -1)

                # opens and pre-processes right image.
                right_path = get_path(basePath, line[2])
                right_image = cv2.imread(right_path)
                right_image = pre_process_image(right_image)

                # augment angle for right by subtracting off-center correction.
                right_measurement = measurement - correction

                # add right image to batch.
                images.append(right_image)
                angles.append(right_measurement)

                # copies can flips right image. flipped images with augmented angle to batch.
                images.append(cv2.flip(right_image.copy(), 1))
                angles.append(right_measurement * -1)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def build_model_graph():
    model = Sequential()

    # Normalization layer
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160, 320, 3)))

    # Cropping layer to reduce noise
    model.add(Cropping2D(cropping=((75, 25), (0, 0))))

    # Convolutional layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Fully connected layers with relu activations
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # compile model with loss function mean squared error for a regression problem.
    # uses adam optimizer.
    model.compile(loss='mse', optimizer='adam')
    return model


if __name__ == '__main__':
    samples = []

    # store all of the steering angles and image locations in samples var.
    with open('data/training/drive/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    # split samples into 20% validation 80% train.
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # create generators for training and validation.
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # early stopping callback to determine when to stop training.
    callback = EarlyStopping(monitor='val_loss',
                             min_delta=0,
                             patience=0,
                             verbose=0, mode='auto')

    # get the model with random weights for training.
    model = build_model_graph()

    # train the model.
    history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples) / 32,
                                         validation_data=validation_generator,
                                         validation_steps=len(validation_samples) / 32, epochs=50, verbose=1,
                                         callbacks=[callback])

    # plot a visualization to compare training loss validation loss per epoch.
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    # save model as model.h5
    print('saving model')
    model_name = 'model.h5'
    model.save(model_name)
    print('model saved {}'.format(model_name))
