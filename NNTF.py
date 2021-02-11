import numpy as np
import snapshots
import measures
import pickle


def create_transitions_matrices(predictions, num_classes=10):
    iterations = list(predictions.keys())
    delta = {}

    for iter in range(len(iterations)-1):
        i_pred = predictions[iterations[iter]]['predictions']
        j_pred = predictions[iterations[iter+1]]['predictions']
        delta_i = np.zeros((num_classes, num_classes))
        for i in range(len(i_pred)):
            delta_i[i_pred[i], j_pred[i]] = delta_i[i_pred[i], j_pred[i]]+1
        delta[iterations[iter]] = delta_i
    return delta


def create_distance_matrix(deltas):
    n_deltas = len(deltas.keys())
    deltas_elem = list(deltas.keys())
    diff_matrix = np.zeros((n_deltas-1, n_deltas-1))

    for i in range(n_deltas - 1):
        for j in range(n_deltas - 1):
            delta_i = deltas[deltas_elem[i]]
            delta_j = deltas[deltas_elem[j]]
            num = abs(delta_i-delta_j)
            den = delta_i+delta_j
            np.fill_diagonal(num, 0)
            np.fill_diagonal(den, 0)
            diff_matrix[i, j] = num.sum()/den.sum()

    return diff_matrix


class NNTF_measures:
    def __init__(self, model_name='lenet', dataset='cifar10', lr=0.001, mt=0.9, wd=5e-3,
                 max_epochs=100, batch_size=128, k_steps=1, l_min=2, lam_a=0.5, lam_b=0.7, ent_a=0.5, ent_b=0.7):
        self.model_name = model_name
        self.dataset = dataset
        self.lr = lr
        self.mt = mt
        self.wd = wd

        self.l_min = l_min
        self.lam_a = lam_a
        self.lam_b = lam_b
        self.ent_a = ent_a
        self.ent_b = ent_b

        # training the model
        model = snapshots.models_training(model_type=model_name, dataset=dataset, learning_rate=lr,
                                          momentum=mt, weight_decay=wd,
                                          maxepoches=max_epochs, batch_size=batch_size, k_steps=k_steps)
        pickle.dump(model.snapshots_info, open(
            f'iterations_info/{model.dataset}_{model.model_type}_lr_{model.learning_rate}_mnt_{model.momentum}_wd_{model.weight_decay}.pickle',
            'wb'))
        self.num_classes = model.num_classes

        self.distance_matrix()
        self.measures = self.compute_rqa()

    def compute_rqa(self):
        # computing RQA measures
        dist_matrix = pickle.load(open(
            f'dist_matrix/{self.dataset}_{self.model_name}_lr_{self.lr}_mnt_{self.mt}_wd_{self.wd}.pickle',
            'rb'))
        res = measures.RQA(dist_matrix, self.l_min, self.lam_a, self.lam_b, self.ent_a, self.ent_b)
        pickle.dump(dist_matrix, open(
            f'RQA_measures/{self.dataset}_{self.model_name}_lr_{self.lr}_mnt_{self.mt}_wd_{self.wd}.pickle',
            'wb'))

        return res

    def distance_matrix(self):
        # creating distance matrices
        snapshots_info = pickle.load(open(
            f'iterations_info/{self.dataset}_{self.model_name}_lr_{self.lr}_mnt_{self.mt}_wd_{self.wd}.pickle',
            'rb'))
        delta = create_transitions_matrices(snapshots_info, self.num_classes)
        dist_matrix = create_distance_matrix(delta)
        pickle.dump(dist_matrix, open(
            f'dist_matrix/{self.dataset}_{self.model_name}_lr_{self.lr}_mnt_{self.mt}_wd_{self.wd}.pickle',
            'wb'))


if __name__ == '__main__':
    # model = snapshots.models_training(model_type='lenet', dataset='cifar10', learning_rate=0.001, momentum=0.9, weight_decay=5e-4,
    #              maxepoches=80, batch_size=128, k_steps=5)
    # pickle.dump(model.snapshots_info, open(
    #     f'iterations_info/{model.dataset}_{model.model_type}_lr_{model.learning_rate}_mnt_{model.momentum}_wd_{model.weight_decay}.pickle',
    #     'wb'))

    snapshots_info = pickle.load(open(
        f'iterations_info/cifar10_lenet_lr_0.001_mnt_0.9_wd_0.0005.pickle',
        'rb'))
    delta = create_transitions_matrices(snapshots_info)
    dist_matrix = create_distance_matrix(delta)
    pickle.dump(dist_matrix, open(
        f'dist_matrix/cifar10_lenet_lr_0.001_mnt_0.9_wd_0.0005.pickle',
        'wb'))