import concurrent.futures
import time
import random
import numba
import numpy as np
import math
from multiprocessing import Pool, Manager, Process

from tqdm import tqdm

from utils import dtw
from utils import dpc


@numba.jit(nopython=True)
def update_loss(v, x, u, c, n, dtw_path, lamda, m, q):
    """
    Update the loss of objective function
    """
    new_loss = 0
    for i in range(c):
        v_i = v[i]
        for j in range(n):
            x_j = x[j]
            path = dtw_path[i][j]
            for k in range(path.shape[1]):
                dist = np.power(v_i[:, path[0, k]] - x_j[:, path[1, k]], 2)
                sum_dist = np.sum(np.multiply(np.power(lamda, q), dist))
                if u[i, j] == 0:
                    u[i, j] += 0.0001
                new_loss += sum_dist * np.power(u[i, j], m)
    return new_loss


@numba.jit(nopython=True)
def restore_padded_matrix(padded_matrix):
    first_negative_column = np.argmax(padded_matrix == -1, axis=1).min()
    restored_matrix = padded_matrix[:, :first_negative_column]
    return restored_matrix


@numba.jit(nopython=True)
def update_v(v, x, u, c, n, D, dtw_path, m):
    """
    Update cluster center
    """
    new_v = []
    for i in range(c):
        b = v[i].shape[1]
        numer_sum = np.zeros((D, b))
        denom_sum = np.zeros(b)

        for j in range(n):
            path = restore_padded_matrix(dtw_path[i][j])
            x_j = x[j]
            for k in range(path.shape[1]):
                if u[i, j] == 0:
                    u[i, j] = 0.0001
                numer_sum[:, path[0, k]] += x_j[:, path[1, k]] * np.power(u[i, j], m)
                denom_sum[path[0, k]] += np.power(u[i, j], m)

        new_vi = numer_sum / denom_sum
        new_v.append(new_vi)
    return new_v


@numba.jit(nopython=True)
def update_lamda(v, x, u, c, n, D, dtw_path, m, q):
    """
    Update dimension weights
    """
    A = []
    for s in range(D):
        Ad = 0
        for i in range(c):
            for j in range(n):
                path = restore_padded_matrix(dtw_path[i][j])
                for k in range(path.shape[1]):
                    if u[i, j] == 0:
                        u[i, j] = 0.0001
                    # Ad += pow(u[i, j], m) * pow(v[i][s, path[0, k]] - x[j][s, path[1, k]], 2)
                    Ad += np.power(u[i, j], m) * np.power(v[i][s, path[0, k]] - x[j][s, path[1, k]], 2)
        A.append(Ad)

    new_lamda = np.zeros(D)
    for d in range(D):
        denom_sum = 0
        for s in range(D):
            if A[d] == 0:
                A[d] = 0.0001
            denom_sum += pow(A[d] / (A[s] + 1), 1 / (q - 1))
        new_lamda[d] = 1 / denom_sum
        if new_lamda[d] > 100 or new_lamda[d] == 0:
            new_lamda[d] = 1e-6

    return new_lamda


@numba.jit(nopython=True)
def update_dtw(c, n, v, x, lamda, q):
    """
    Update DTW distance and OWPs
    """
    new_dist = np.zeros((c, n))
    new_owp = []
    for i in range(c):
        tmp = []
        for j in range(n):
            new_dist[i, j], path = dtw.get_dtw(t1=v[i], t2=x[j], lamda=lamda, q=q)
            padded_path = -1 * np.ones((2, 30), dtype=path.dtype)
            padded_path[:, :path.shape[1]] = path
            tmp.append(padded_path)
        # tmp = np.stack(tmp)
        new_owp.append(tmp)
    return new_dist, new_owp


@numba.jit(nopython=True)
def update_u(c, n, dtw_dist, m):
    """
    Update membership degree matrix
    """
    new_u = np.zeros((c, n))
    for i in range(c):
        for j in range(n):
            denom_sum = 0
            is_coincide = [False, 0]

            for s in range(c):
                if dtw_dist[s, j] == 0:
                    is_coincide[0] = True
                    is_coincide[1] = s
                    break
                denom_sum += pow(dtw_dist[i, j] / dtw_dist[s, j], 1 / (m - 1))

            if is_coincide[0]:
                if is_coincide[1] == i:
                    new_u[i, j] = 1
                else:
                    new_u[i, j] = 0
            else:
                new_u[i, j] = 1 / denom_sum
    return new_u


class FcmWdtw:

    def __init__(self, data, c, m, q, max_iter, dc_percent, class_label=None, anom_label=None):
        self.max_iter = max_iter  # maximum iterations
        self.x = data  # input data
        self.c = c  # class number
        self.m = m  # fuzzy order
        self.q = q  # weight order
        self.class_label = class_label  # class label
        self.anom_label = anom_label  # anomaly label
        self.dc_percent = dc_percent  # intercept percentage of DPC
        self.D = self.x[0].shape[0]  # sample dimensions
        self.n = len(self.x)  # dataset size
        self.dtw_path = []  # OWP
        self.dtw_dist = np.zeros((self.c, self.n))  # DTW distance
        self.u = np.ones((self.c, self.n)) / self.c  # membership degree matrix
        self.lamda = np.ones(self.D) / self.D  # dimension weights
        self.v = None  # cluster centers
        self.loss = math.inf  # loss of objective function
        print("dataset info: dimension:", self.D, " class:", self.c, " size:", self.n)

    def dpc_initiate(self):
        """
        Initialize cluster centers
        """
        dpc_centers = dpc.get_dpc(self.x, lamda=self.lamda, c=self.c, percent=self.dc_percent)
        centers = []
        for ind in dpc_centers:
            centers.append(self.x[ind])
        return centers

    def random_initiate(self):
        # 从样本中随机选择c个作为簇中心
        return random.sample(self.x, self.c)

    # @numba.jit(nopython=False)
    # def update_lamda(self):
    #     """
    #     Update dimension weights
    #     """
    #     A = []
    #     for s in range(self.D):
    #         Ad = 0
    #         for i in range(self.c):
    #             for j in range(self.n):
    #                 path = self.dtw_path[i][j]
    #                 for k in range(path.shape[1]):
    #                     if self.u[i, j] == 0:
    #                         self.u[i, j] = 0.0001
    #                     Ad += pow(self.u[i, j], self.m) * \
    #                           pow(self.v[i][s, path[0, k]] - self.x[j][s, path[1, k]], 2)
    #         A.append(Ad)
    #
    #     new_lamda = np.zeros(self.D)
    #     for d in range(self.D):
    #         denom_sum = 0
    #         for s in range(self.D):
    #             if A[d] == 0:
    #                 A[d] = 0.0001
    #             denom_sum += pow(A[d] / (A[s] + 1), 1 / (self.q - 1))
    #         new_lamda[d] = 1 / denom_sum
    #         if new_lamda[d] > 100 or new_lamda[d] == 0:
    #             new_lamda[d] = 1e-6
    #
    #     return new_lamda

    def update_v(self):
        """
        Update cluster center
        """
        new_v = []
        for i in range(self.c):
            b = self.v[i].shape[1]
            numer_sum = np.zeros((self.D, b))
            denom_sum = np.zeros(b)

            for j in range(self.n):
                path = self.dtw_path[i][j]
                x_j = self.x[j]
                for k in range(path.shape[1]):
                    if self.u[i, j] == 0:
                        self.u[i, j] = 0.0001
                    numer_sum[:, path[0, k]] += x_j[:, path[1, k]] * pow(self.u[i, j], self.m)
                    denom_sum[path[0, k]] += pow(self.u[i, j], self.m)

            new_vi = numer_sum / denom_sum
            new_v.append(new_vi)
        return new_v

    # def update_v(self):
    #     """
    #     Update cluster center using multithreading
    #     """
    #     new_v = []
    #     pool = Pool(100)
    #
    #     manager = Manager()
    #     for i in range(self.c):
    #         # print('bbbbbbb')
    #         result = pool.apply_async(func=process_center_i,
    #                          args=(self.v, self.x, self.D, self.n, self.dtw_path, self.u, self.m, i))
    #         new_v.append(result.get())
    #
    #     pool.close()
    #     pool.join()
    #     return new_v

    # def update_loss(self):
    #     """
    #     Update the loss of objective function
    #     """
    #     new_loss = 0
    #     for i in range(self.c):
    #         v_i = self.v[i]
    #         for j in range(self.n):
    #             x_j = self.x[j]
    #             path = self.dtw_path[i][j]
    #             for k in range(path.shape[1]):
    #                 dist = np.power(v_i[:, path[0, k]] - x_j[:, path[1, k]], 2)
    #                 sum_dist = np.sum(np.multiply(np.power(self.lamda, self.q), dist))
    #                 if self.u[i, j] == 0:
    #                     self.u[i, j] += 0.0001
    #                 new_loss += sum_dist * pow(self.u[i, j], self.m)
    #     return new_loss

    # def update_loss(self):
    #     """
    #     Update the loss of objective function using multithreading
    #     """
    #     new_loss = 0
    #     pool = Pool(100)
    #     manager = Manager()
    #
    #     for i in range(self.c):
    #         result = pool.apply_async(func=loss_center_i,
    #                                   args=(self.v, self.x, self.lamda, self.q, self.n, self.dtw_path, self.u, self.m, i))
    #         new_loss += result.get()
    #
    #     pool.close()
    #     pool.join()
    #
    #     return new_loss

    def fcm_wdtw(self):
        """
        FCM-wDTW realization
        """
        # initialize cluster centers
        # self.v = self.dpc_initiate()
        self.v = self.random_initiate()
        print("Initialize cluster centers")

        start_time = time.time()
        all_loss = []
        for i in range(self.max_iter):
            print("iteration: ", i)
            self.dtw_dist, self.dtw_path = update_dtw(self.c, self.n, np.stack(self.v, axis=0),
                                                      np.stack(self.x, axis=0), self.lamda, self.q)  # update DTW OWPs
            print("Update OWP")
            self.u = update_u(self.c, self.n, self.dtw_dist, self.m)  # update membership matrix
            print("Update U")
            self.lamda = update_lamda(np.stack(self.v, axis=0), np.stack(self.x, axis=0),
                                      self.u, self.c, self.n, self.D, np.array(self.dtw_path), self.m,
                                      self.q)  # update dimension weights
            print("Update lamda")
            # print(self.lamda)
            self.v = update_v(np.stack(self.v, axis=0), np.stack(self.x, axis=0),
                              self.u, self.c, self.n, self.D, np.array(self.dtw_path), self.m)  # update cluster centers
            print("Update cluster centers")
            # print(self.v)
            loss = update_loss(np.stack(self.v, axis=0), np.stack(self.x, axis=0), self.u, self.c, self.n,
                               np.array(self.dtw_path), self.lamda, self.m, self.q)  # update loss
            if loss > self.loss:
                break
            self.loss = loss
            all_loss.append(loss)
            print("Opt loss: ", loss)

        end_time = time.time()
        time_cost = end_time - start_time

        ri = None
        if self.class_label is not None:
            ri = self.cal_ri(np.argmax(self.u, axis=0))

        return ri, all_loss, time_cost

    def anomaly_score(self, x, y, dist_measure):
        """
        Anomaly score
        """
        if dist_measure == 'ED':
            return ((x - y) ** 2).sum()
        elif dist_measure == 'WED':
            return (np.power(self.lamda, self.q) * ((x - y) ** 2).sum(axis=1)).sum()
        elif dist_measure == 'WDTW':
            return dtw.get_dtw(t1=x, t2=y, lamda=self.lamda, q=self.q)[0]

    def anomaly_detect(self):
        """
        Reconstruct samples and compute anomaly score
        """
        re_samples = []
        ed_scores = []
        wed_scores = []
        wdtw_scores = []
        for j in range(self.n):
            a = self.x[j].shape[1]
            numer_sum = np.zeros((self.D, a))
            denom_sum = np.zeros(a)

            for i in range(self.c):
                path = self.dtw_path[i][j]
                v_i = self.v[i]
                for k in range(path.shape[1]):
                    numer_sum[:, path[1, k]] += v_i[:, path[0, k]] * pow(self.u[i, j], self.m)
                    denom_sum[path[1, k]] += pow(self.u[i, j], self.m)

            re_xj = numer_sum / denom_sum
            re_samples.append(re_xj)

            ed_scores.append(self.anomaly_score(re_xj, self.x[j], 'ED'))
            wed_scores.append(self.anomaly_score(re_xj, self.x[j], 'WED'))
            wdtw_scores.append(self.anomaly_score(re_xj, self.x[j], 'WDTW'))

        scores = {'ed_scores': ed_scores, 'wed_scores': wed_scores, 'wdtw_scores': wdtw_scores}

        # anomaly number
        self.anom_label[-20:] = 1
        anom_num = int(self.anom_label.sum())
        anomaly = {'anom_ed': np.argsort(ed_scores)[-1 * anom_num:],
                   'anom_wed': np.argsort(wed_scores)[-1 * anom_num:],
                   'anom_wdtw': np.argsort(wdtw_scores)[-1 * anom_num:]}

        metric_ed = self.cal_F1(anomaly['anom_ed'])
        metric_wed = self.cal_F1(anomaly['anom_wed'])
        metric_wdtw = self.cal_F1(anomaly['anom_wdtw'])

        return anomaly, scores, metric_ed, metric_wed, metric_wdtw

    def cal_ri(self, y_pred):
        """
        Compute Rand Index
        """
        n = len(self.class_label)
        a, b = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if (self.class_label[i] == self.class_label[j]) & (y_pred[i] == y_pred[j]):
                    a += 1
                elif (self.class_label[i] != self.class_label[j]) & (y_pred[i] != y_pred[j]):
                    b += 1
                else:
                    pass
        ri = (a + b) / (n * (n - 1) / 2)
        return ri

    def cal_F1(self, pred):
        """
        Compute precision, recall, F1
        """
        gt = np.nonzero(self.anom_label)[0]
        precision = len(set(pred) & set(gt)) / len(pred)
        recall = len(set(pred) & set(gt)) / len(gt)
        F1 = 2 * precision * recall / (precision + recall)
        metric = {'precision': [precision], 'recall': [recall], 'F1': [F1]}
        return metric
