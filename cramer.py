#!bin/python3

import numpy as np


def mxw(a, b):
    nw, nk = a.shape
    z = np.zeros((nw,))
    for i in range(nw):
        s = 0.0
        for k in range(nk):
            s += a[i, k] * b[k]
        z[i] = s
    z = z.reshape(nk, 1)
    return z


def cramer(a, b):
    det = np.linalg.det(a)
    x = np.zeros((len(a), 1))

    for i in range(len(a)):
        copied = a[:, i]
        copied = copied.copy()
        a[:, i] = b[:, 0]
        x[i, 0] = (np.linalg.det(a) / det)
        a[:, i] = copied
    return x


def generator(size, diag, up, down):
    A = np.zeros((size, size))

    B = np.zeros((size, 1))
    for i in range(size):
        for k in range(size):
            A[i, i] = diag
            for j in range(i + 1, size):
                A[i, j] = up
            for j in range(i):
                A[i, j] = down

        B[i, 0] = 1.0

    return A, B

