#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 18 22:14:30 2023
@author: cai
"""
import pandas as pd

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


def plot_embedding(distance, gt_label, fcm_label):

    embeddings = TSNE(n_components=2, metric="precomputed", random_state=0).fit_transform(distance)

    x_min, x_max = np.min(embeddings, 0), np.max(embeddings, 0)
    data = (embeddings - x_min) / (x_max - x_min)

    _font_size = 8

    plt.subplot(2, 4, 1)

    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='.', c=plt.cm.Set1(gt_label[i]))

    plt.xticks([])
    plt.yticks([])
    plt.title("Groud Truth", fontsize=_font_size)

    plt.subplot(2, 4, 2)

    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='.', c=plt.cm.Set1(fcm_label[i]))

    plt.xticks([])
    plt.yticks([])
    plt.title("FCM-wDTW", fontsize=_font_size)

    plt.subplot(2, 4, 3)

    fig = plt.gcf()
    fig.set_size_inches(6, 5)
    fig.savefig("Emedding_For_Algos.pdf", format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    distance = pd.read_excel('../../result/ECG.xlsx', sheet_name='wDTW samples_centers', index_col=0)
    pt_matrix = pd.read_excel('../../result/ECG.xlsx', sheet_name='membership matrix', index_col=0)
    gt_label = pd.read_excel('../../result/ECG.xlsx', sheet_name='gt_label', index_col=0)

    distance = distance.T
    pt_matrix = pt_matrix.T
    gt_label = gt_label[0]
    fcm_label = pt_matrix.idxmax(1)

    plot_embedding(distance, gt_label, fcm_label)
