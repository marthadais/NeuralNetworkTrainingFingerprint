from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import models
import datasets


class models_training:
    def __init__(self, model_type='lenet', dataset='cifar10', learning_rate=0.001, momentum=0.9, weight_decay=5e-4,
                 maxepoches=100, batch_size=128, k_steps=1):
        self.timestamp = {}
        self.loss = {}
        self.acc = {}

        self.model_type = model_type
        self.dataset = dataset
        self.maxepoches = maxepoches
        self.batch_size = batch_size
        self.k = k_steps

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum

        if self.dataset == 'cifar10':
            self.x_train, self.y_train, self.num_classes, self.x_shape = datasets.get_cifar10_data()
        else:
            self.x_train, self.y_train, self.num_classes, self.x_shape = datasets.get_mnist_data()

        if self.model_type == 'lenet':
            self.model = models.build_lenet(self.x_shape, self.num_classes, self.weight_decay)
        else:
            self.model = models.build_vgg(self.x_shape, self.num_classes, self.weight_decay)

        self.train()

    def train(self, verbose=True):

        # avoid load all the dataset on the RAM memory (slow processing)
        # it is also used for data augmentation
        data_generator = ImageDataGenerator()
        data_generator.fit(self.x_train)

        # optimization details
        sgd = optimizers.SGD(lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        if verbose:
            print(f'Running {self.model_type.upper()} model with {self.dataset.upper()} dataset...')
            print(f'Learning Rate: {self.learning_rate}')
            print(f'Momentum: {self.momentum}')
            print(f'Weight Decay: {self.weight_decay}')

        # save the predictions in every k epochs
        for i in range(0, self.maxepoches, self.k):
            historytemp = self.model.fit(data_generator.flow(self.x_train, self.y_train, batch_size=self.batch_size,
                                                             shuffle=True),
                                         steps_per_epoch=self.x_train.shape[0] // self.batch_size, epochs=self.k)
            predicted_x = self.model.predict(self.x_train)
            self.timestamp[i] = predicted_x.argmax(axis=1)
            self.loss[i] = historytemp.history['loss']
            self.acc[i] = historytemp.history['accuracy']

        self.model.save_weights(f'models/{self.model_type}.h5')


if __name__ == '__main__':
    model = models_training(model_type='vgg')
    # model = models_training(model_type='lenet')