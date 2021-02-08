from keras import optimizers
import models
import datasets

class models_training:
    def __init__(self, model_type='lenet', dataset='cifar10', learning_rate=0.001, momentum=0.9, weight_decay=5e-4):
        self.timestamp = {}
        self.loss = {}
        self.acc = {}

        self.model_type = model_type
        self.dataset = dataset
        self.num_classes = 10
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum

        if self.dataset == 'cifar10':
            self.x_train, self.y_train, self.x_test, self.y_test, self.x_shape = datasets.get_cifar10_data()
        else:
            self.x_train, self.y_train, self.x_test, self.y_test, self.x_shape = datasets.get_mnist_data()

        if self.model_type == 'lenet':
            self.model = models.build_lenet(self.x_shape, self.num_classes, self.weight_decay)
        else:
            self.model = models.build_vgg(self.x_shape, self.num_classes, self.weight_decay)

        self.model = self.train()



    def train(self):

        # training parameters
        batch_size = 128
        maxepoches = 10
        k = 1

        # optimization details
        sgd = optimizers.SGD(lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # save the predictions
        for i in range(maxepoches):
            # historytemp = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
            historytemp = self.model.fit(self.x_train, self.y_train, batch_size=batch_size,
                                         steps_per_epoch=self.x_train.shape[0] // batch_size, epochs=k)
            # validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)
            predicted_x = self.model.predict(self.x_train)
            self.timestamp[i] = predicted_x.argmax(axis=1)
            self.loss[i] = historytemp.history['loss']
            self.acc[i] = historytemp.history['accuracy']

        self.model.save_weights(f'models/{self.model_type}.h5')


if __name__ == '__main__':
    model = models_training()