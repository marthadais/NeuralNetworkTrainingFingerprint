import keras
from keras.datasets import cifar10, mnist


def get_mnist_data():
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_shape = [28, 28, 1]

    return x_train, y_train, x_test, y_test, num_classes, x_shape


def get_cifar10_data():
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_shape = [32, 32, 3]

    return x_train, y_train, x_test, y_test, num_classes, x_shape
