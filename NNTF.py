from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import models
import datasets
import pickle
import measures
import differences as df
import os


class NNTF:
    def __init__(self, model_type='lenet', dataset='cifar10', learning_rate=0.001, momentum=0.9, weight_decay=5e-4,
                 max_epochs=100, batch_size=128, k_steps=1, l_min=1, interval=0.05):
        self.snapshots_info = {}

        self.model_type = model_type
        self.dataset = dataset
        self.maxepoches = max_epochs
        self.batch_size = batch_size
        self.k = k_steps
        self.num_classes = 10

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.l_min = l_min
        self.interval = interval
        self.output_file = f'{self.dataset}_{self.model_type}_lr_{self.learning_rate}_mnt_{self.momentum}_wd_{self.weight_decay}.pickle'

        if not os.path.isfile(f'snapshots_info/{self.output_file}'):
            self.train()

        self.distance_matrix()
        self.measures = self.compute_rqa()

    def train(self, verbose=True):
        if self.dataset == 'cifar10':
            x_train, y_train, x_test, y_test, self.num_classes, x_shape = datasets.get_cifar10_data()
        else:
            x_train, y_train, x_test, y_test, self.num_classes, x_shape = datasets.get_mnist_data()

        if self.model_type == 'lenet':
            model = models.build_lenet(x_shape, self.num_classes, self.weight_decay)
        else:
            model = models.build_vgg16(x_shape, self.num_classes, self.weight_decay)

        if verbose:
            print(f'Running {self.model_type.upper()} model with {self.dataset.upper()} dataset...')
            print(f'It will save at every {self.k} epochs...')
            print(f'Learning Rate: {self.learning_rate}')
            print(f'Momentum: {self.momentum}')
            print(f'Weight Decay: {self.weight_decay}')

        # avoid load all the dataset on the RAM memory (slow processing)
        # it is also used for data augmentation
        data_generator = ImageDataGenerator()
        data_generator.fit(x_train)

        # optimization details
        sgd = optimizers.SGD(lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # save the predictions in every k epochs
        snapshots_info = {}
        for i in range(0, self.maxepoches, self.k):
            if verbose:
                print(f'Running {i} of {self.maxepoches}...')
            history_info = model.fit(data_generator.flow(x_train, y_train, batch_size=self.batch_size, shuffle=True),
                                         steps_per_epoch=x_train.shape[0] // self.batch_size, epochs=self.k,
                                          verbose=0)
            predicted_x = model.predict(x_train)
            snapshots_info[i] = {}
            snapshots_info[i]['predictions'] = predicted_x.argmax(axis=1)
            snapshots_info[i]['loss'] = history_info.history['loss'][self.k - 1]
            snapshots_info[i]['accuracy'] = history_info.history['accuracy'][self.k - 1]
            snapshots_info[i]['test_score'] = model.evaluate(x_test, y_test, verbose=0)
            if verbose:
                print(f"Loss and Acc train: {snapshots_info[i]['loss']} {snapshots_info[i]['accuracy']}")
                print(f"Loss and Acc test: {snapshots_info[i]['test_score']}")

        model.save_weights(f'models/{self.output_file}.h5')
        pickle.dump(snapshots_info, open(f'snapshots_info/{self.output_file}', 'wb'))

        return snapshots_info

    def compute_rqa(self):
        # computing RQA measures
        dist_matrix = pickle.load(open(f'dist_matrix/{self.output_file}', 'rb'))
        res = measures.RQA(dist_matrix, self.l_min, self.interval)
        pickle.dump(res.all_measures, open(f'RQA_measures/{self.output_file}', 'wb'))

        return res

    def distance_matrix(self):
        # creating distance matrices
        snapshots = pickle.load(open(
            f'snapshots_info/{self.output_file}', 'rb'))
        delta = df.create_transitions_matrices(snapshots, self.num_classes)
        dist_matrix = df.create_distance_matrix(delta)
        pickle.dump(dist_matrix, open(
            f'dist_matrix/{self.output_file}', 'wb'))

        return dist_matrix

