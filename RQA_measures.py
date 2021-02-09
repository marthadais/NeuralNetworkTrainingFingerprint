import numpy as np


def split_list(x, pos):
    my_list = list(x)
    pos_b = pos
    while pos_b < len(my_list) and my_list[pos_b] == 0:
        pos_b = pos_b + 1
    return np.array(my_list[:pos]), np.array(my_list[pos_b:])


def get_diagonals(matrix, lmin=2):

    diags = [matrix.diagonal(i) for i in range(matrix.shape[1]-1, -matrix.shape[1], -1)]

    res = []
    for diag in diags:
        if diag.sum() != len(diag):
            b = diag
            while len(b) != 0 and b.sum() != 0 and b.sum() != len(b):
                a, b = split_list(b, np.argmin(b))
                if len(a) >= lmin and a.sum() > 0:
                    res.append(a)
            if len(b) >= lmin and b.sum() > 0:
                res.append(b)
        else:
            if len(diag) >= lmin and diag.sum() > 0:
                res.append(diag)

    return res


def get_diagonals_lenght(matrix, lmin=2):

    diags = [matrix.diagonal(i) for i in range(matrix.shape[1]-1, -matrix.shape[1], -1)]

    res = []
    for diag in diags:
        if diag.sum() != len(diag):
            b = diag
            while len(b) != 0 and b.sum() != 0 and b.sum() != len(b):
                a, b = split_list(b, np.argmin(b))
                if len(a) >= lmin and a.sum() > 0:
                    res.append(len(a))
            if len(b) >= lmin and b.sum() > 0:
                res.append(len(b))
        else:
            if len(diag) >= lmin and diag.sum() > 0:
                res.append(len(diag))

    return res

def get_verticals(matrix, lmin=2):

    verticals = [matrix[:, i] for i in range(matrix.shape[1])]

    res = []
    for v in verticals:
        if v.sum() != len(v):
            b = v
            while len(b) != 0 and b.sum() != 0 and b.sum() != len(b):
                a, b = split_list(b, np.argmin(b))
                if len(a) >= lmin and a.sum() > 0:
                    res.append(a)
            if len(b) >= lmin and b.sum() > 0:
                res.append(b)
        else:
            if len(v) >= lmin and v.sum() > 0:
                res.append(v)
    return res

def get_verticals_lenght(matrix, lmin=2):

    verticals = [matrix[:, i] for i in range(matrix.shape[1])]

    res = []
    for v in verticals:
        if v.sum() != len(v):
            b = v
            while len(b) != 0 and b.sum() != 0 and b.sum() != len(b):
                a, b = split_list(b, np.argmin(b))
                if len(a) >= lmin and a.sum() > 0:
                    res.append(len(a))
            if len(b) >= lmin and b.sum() > 0:
                res.append(len(b))
        else:
            if len(v) >= lmin and v.sum() > 0:
                res.append(len(v))

    return res


def entropy(matrix, lmin=2):
    res = get_diagonals_lenght(matrix, lmin)
    hist = np.histogram(res, bins=range(lmin - 1, len(matrix.diagonal()) + 2))
    probs = hist[0] / np.sum(res)+1e-8

    return - np.sum(probs*np.log(probs+1e-8))


def laminarity(matrix, lmin=2):

    v1 = get_verticals_lenght(matrix, 1)
    v_lmin = get_verticals_lenght(matrix, lmin)
    hist1 = np.histogram(v1, bins=range(0, matrix.shape[1] + 1))
    hist_lmin = np.histogram(v_lmin, bins=range(0, matrix.shape[1] + 1))

    v_lenght = list(range(0, matrix.shape[1]))

    return np.sum(v_lenght*hist1[0])/np.sum(v_lenght*hist_lmin[0])


if __name__ == '__main__':
    matrix = np.array(
         [[0, 1, 1, 1, 0],
          [1, 0, 1, 0, 1],
          [0, 1, 0, 1, 1],
          [1, 0, 1, 0, 1],
          [1, 1, 1, 1, 0]])
    lmin=3

    L = laminarity(matrix, lmin)
    H = entropy(matrix, lmin)
    # diags = get_diagonals(matrix)
    # verticals = get_verticals(matrix)
    # res = get_diagonals_lenght(matrix, lmin)
    # hist = np.histogram(res, bins=range(lmin-1, len(matrix.diagonal())+1))
    # probs = hist[0] / np.sum(res)
