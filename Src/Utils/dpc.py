import numpy as np
from utils import dtw


def getDistanceMatrix(data, lamda):
    """
    Compute DTW distance
    """
    n = len(data)
    dist_mat = np.ones((n, n)) * 1e100
    for i in range(n):
        print('DTW matrix for DPC: ', i)
        for j in range(i, n):
            dist_mat[i, j] = dist_mat[j, i] = dtw.get_dtw(t1=data[i], t2=data[j], lamda=lamda, q=1)[0]
    return dist_mat


def select_dc(dists, percent):
    """
    Compute the dense threshold
    """
    N = np.shape(dists)[0]
    tt = np.reshape(dists, N * N)
    position = int(N * (N - 1) * percent / 100)
    dc = np.sort(tt)[position + N]
    return dc


def get_density(dists, dc, method=None):
    """
    Compute the local density of each sample
    """
    N = np.shape(dists)[0]
    rho = np.zeros(N)
    for i in range(N):
        if method is None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho


def get_deltas(dists, rho):
    """
    Compute the distance for each sample to the nearest sample with higher density
    """
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        if i == 0:
            deltas[index_rho[0]] = np.max(dists[index, :])
            continue
        index_higher_rho = index_rho[:i]
        deltas[index] = np.min(dists[index, index_higher_rho])
    return deltas


def get_centers(rho, deltas, c):
    """
    Pick samples with largest rho*delta as cluster centers
    """
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:c]


def get_dpc(data, lamda, c, percent):
    """
    Density peak cluster
    """
    dists = getDistanceMatrix(data=data, lamda=lamda)
    dc = select_dc(dists, percent=percent)
    rho = get_density(dists, dc)
    deltas = get_deltas(dists, rho)
    centers = get_centers(rho, deltas, c=c)
    print("dpc centers:", centers)
    return centers
