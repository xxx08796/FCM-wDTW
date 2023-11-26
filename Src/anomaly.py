#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import concurrent.futures

import numpy as np

sys.path.append("..")
import pandas as pd
from utils import fcmwdtw
# from src.utils import fcmwdtw
from scipy.io import loadmat



dataset = 'PCSO20'
path = '../data/' + dataset + '/poly-channels-single-of-20.mat'
raw_data = loadmat(path)
X = raw_data["X_test"]
Y = raw_data["Y_test"]
label = Y.reshape(-1)
data = list(X[0])

print("Cluster " + dataset)

# set parameters
para = [(10, 1.5, -5), (20, 1.5, -5), (30, 1.5, -5), (40, 1.5, -5), (50, 1.5, -5),
        (10, 3, -5), (20, 3, -5), (30, 3, -5), (40, 3, -5), (50, 3, -5),
        (10, 1.5, -2), (20, 1.5, -2), (30, 1.5, -2), (40, 1.5, -2), (50, 1.5, -2),
        (10, 3, 2), (20, 3, 2), (30, 3, 2), (40, 3, 2), (50, 3, 2),
        (10, 1.5, 5), (20, 1.5, 5), (30, 1.5, 5), (40, 1.5, 5), (50, 1.5, 5),
        (10, 3, 5), (20, 3, 5), (30, 3, 5), (40, 3, 5), (50, 3, 5)]
# clus_num = 15
# m = 1.5
# q = -5
sys.argv = [0,'0']
clus_num, m, q = para[int(sys.argv[1])]
# clus_num, m, q = para[int(sys.argv[1])]
dc_intercept = 10
iteration = 20

writer = pd.ExcelWriter('./result/' + dataset + sys.argv[1] + '.xlsx')

# cluster
opt = fcmwdtw.FcmWdtw(data=data, c=clus_num, m=m, q=q, max_iter=iteration,
                      dc_percent=dc_intercept, anom_label=label)
result = opt.fcm_wdtw()
print("rand index:", result)

# anomaly detect
anomaly, scores, metric_ed, metric_wed, metric_wdtw = opt.anomaly_detect()

pd.DataFrame(opt.dtw_dist).to_excel(writer, sheet_name='wDTW samples_centers')
pd.DataFrame(opt.u).to_excel(writer, sheet_name='membership matrix')
pd.DataFrame(opt.lamda).to_excel(writer, sheet_name='dim weights')
pd.DataFrame(result[1]).to_excel(writer, sheet_name='loss')
pd.DataFrame(opt.anom_label).to_excel(writer, sheet_name='gt_label')
pd.DataFrame(anomaly).to_excel(writer, sheet_name='anomaly')
pd.DataFrame(scores).to_excel(writer, sheet_name='scores')
pd.DataFrame(metric_ed).to_excel(writer, sheet_name='metric_ed')
pd.DataFrame(metric_wed).to_excel(writer, sheet_name='metric_wed')
pd.DataFrame(metric_wdtw).to_excel(writer, sheet_name='metric_wdtw')
for i in range(clus_num):
    pd.DataFrame(opt.v[i]).to_excel(writer, sheet_name='centers' + str(i))

writer.save()
