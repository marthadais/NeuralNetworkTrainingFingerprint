import measures
import distances
import snapshots
import pickle

#training the model
model = snapshots.models_training(model_type='lenet', dataset='cifar10', learning_rate=0.001, momentum=0.9, weight_decay=5e-4,
                 maxepoches=80, batch_size=128, k_steps=5)
pickle.dump(model.snapshots_info, open(
    f'iterations_info/{model.dataset}_{model.model_type}_lr_{model.learning_rate}_mnt_{model.momentum}_wd_{model.weight_decay}.pickle',
    'wb'))

# creating distance matrices
snapshots_info = pickle.load(open(
    f'iterations_info/cifar10_lenet_lr_0.001_mnt_0.9_wd_0.0005.pickle',
    'rb'))
delta = distances.create_transitions_matrices(snapshots_info)
dist_matrix = distances.create_distance_matrix(delta)
pickle.dump(dist_matrix, open(
    f'dist_matrix/cifar10_lenet_lr_0.001_mnt_0.9_wd_0.0005.pickle',
    'wb'))

#computing RQA measures
lmin = 2
b_a = 0.5
b_b = 0.7

dist_matrix = pickle.load(open(
    f'dist_matrix/cifar10_lenet_lr_0.001_mnt_0.9_wd_0.0005.pickle',
    'rb'))

res = measures.RQA(dist_matrix, lmin, b_a, b_b)