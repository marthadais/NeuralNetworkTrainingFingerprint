import numpy as np
import pickle


def split_list(x, pos):
    my_list = list(x)
    pos_b = pos
    while pos_b < len(my_list) and my_list[pos_b] == 0:
        pos_b = pos_b + 1
    return np.array(my_list[:pos]), np.array(my_list[pos_b:])

def count_lines(lines, lmin):
    res = []
    for line in lines:
        if line.sum() != len(line):
            b = line
            while len(b) != 0 and b.sum() != 0 and b.sum() != len(b):
                a, b = split_list(b, np.argmin(b))
                if len(a) >= lmin and a.sum() > 0:
                    res.append(len(a))
            if len(b) >= lmin and b.sum() > 0:
                res.append(len(b))
        else:
            if len(line) >= lmin and line.sum() > 0:
                res.append(len(line))
    return res


class RQA:
    def __init__(self, dist_matrix, lmin=2, interval_a=0.5, interval_b=0.7):
        self.lmin = lmin
        self.interval_a = interval_a
        self.interval_b = interval_b
        self.b_matrix = self.binarize_matrix(dist_matrix)
        self.laminarity = self.laminarity_measure(self.b_matrix)
        self.entropy = self.entropy_measure(self.b_matrix)


    def entropy_measure(self, matrix):
        diags = [matrix.diagonal(i) for i in range(matrix.shape[1]-1, -matrix.shape[1], -1)]
        d = count_lines(diags, self.lmin)
        hist = np.histogram(d, bins=range(self.lmin - 1, len(matrix.diagonal()) + 2))
        probs = hist[0] / np.sum(d)+1e-8

        return - np.sum(probs*np.log(probs+1e-8))


    def laminarity_measure(self, matrix):
        verticals = [matrix[:, i] for i in range(matrix.shape[1])]
        v1 = count_lines(verticals, 1)
        v_lmin = count_lines(verticals, self.lmin)

        hist1 = np.histogram(v1, bins=range(0, matrix.shape[1] + 1))
        hist_lmin = np.histogram(v_lmin, bins=range(0, matrix.shape[1] + 1))

        v_lenght = list(range(0, matrix.shape[1]))

        return np.sum(v_lenght*hist1[0])/np.sum(v_lenght*hist_lmin[0])


    def binarize_matrix(self, matrix):
        res = matrix.copy()
        idx = np.where(np.logical_and(matrix >= self.interval_a, matrix <= self.interval_b))
        mask = np.ones(matrix.shape, bool)
        mask[idx] = False
        res[idx] = 1
        res[mask] = 0
        return res


if __name__ == '__main__':
    # b_matrix = np.array(
    #      [[0, 1, 1, 1, 0],
    #       [1, 0, 1, 0, 1],
    #       [0, 1, 0, 1, 1],
    #       [1, 0, 1, 0, 1],
    #       [1, 1, 1, 1, 0]])
    lmin=2
    b_a = 0.5
    b_b = 0.7

    dist_matrix = pickle.load(open(
        f'dist_matrix/cifar10_lenet_lr_0.001_mnt_0.9_wd_0.0005.pickle',
        'rb'))

    measures = RQA(dist_matrix, lmin, b_a, b_b)
