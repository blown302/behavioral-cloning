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


def normalize(img):
    return img / 127.5 - 1


def pre_process_image(img):
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
                correction = .07
                measurement_boost_factor = .05

                # add measurement
                measurement = measurement + measurement * measurement_boost_factor

                #
                center_path = get_path(basePath, line[0])
                center_image = cv2.imread(center_path)
                center_image = pre_process_image(center_image)

                images.append(center_image)
                angles.append(measurement)

                images.append(cv2.flip(center_image.copy(), 1))
                angles.append(measurement * -1)

                left_path = get_path(basePath, line[1])
                left_image = cv2.imread(left_path)
                left_image = pre_process_image(left_image)
                left_measurement = measurement + correction

                images.append(left_image)
                angles.append(left_measurement)

                images.append(cv2.flip(left_image.copy(), 1))
                angles.append(left_measurement * -1)

                right_path = get_path(basePath, line[2])
                right_image = cv2.imread(right_path)
                right_image = pre_process_image(right_image)
                right_measurement = measurement - correction

                images.append(right_image)
                angles.append(right_measurement)

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

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


if __name__ == '__main__':
    samples = []
    with open('data/training/drive/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    callback = EarlyStopping(monitor='val_loss',
                             min_delta=0,
                             patience=0,
                             verbose=0, mode='auto')

    model = build_model_graph()

    history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples) / 32,
                                         validation_data=validation_generator,
                                         validation_steps=len(validation_samples) / 32, epochs=50, verbose=1,
                                         callbacks=[callback])

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    print('saving model')
    model_name = 'model.h5'
    model.save(model_name)
    print('model saved {}'.format(model_name))
