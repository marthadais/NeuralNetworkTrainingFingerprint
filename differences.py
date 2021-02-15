import numpy as np


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