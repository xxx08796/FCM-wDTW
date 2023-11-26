#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd

para = [(10, 1.5, -5), (20, 1.5, -5), (30, 1.5, -5), (40, 1.5, -5), (50, 1.5, -5),
        (10, 3, -5), (20, 3, -5), (30, 3, -5), (40, 3, -5), (50, 3, -5),
        (10, 1.5, -2), (20, 1.5, -2), (30, 1.5, -2), (40, 1.5, -2), (50, 1.5, -2),
        (10, 3, 2), (20, 3, 2), (30, 3, 2), (40, 3, 2), (50, 3, 2),
        (10, 1.5, 5), (20, 1.5, 5), (30, 1.5, 5), (40, 1.5, 5), (50, 1.5, 5),
        (10, 3, 5), (20, 3, 5), (30, 3, 5), (40, 3, 5), (50, 3, 5)]

dataset = ['CalIt2', 'poly-channels-single-of-5', 'poly-channels-single-of-10', 'poly-channels-single-of-20']

for i in range(4):
    filename = dataset[i]
    if i == 0:
        path = '../../result/CalIt2/'
    else:
        path = '../../result/GutenTAG/' + filename + '/'

    result = pd.DataFrame()
    for f in os.listdir(path):
        print(f)
        metric = pd.read_excel(path + f, sheet_name='metric_wdtw', index_col=0)
        if i == 0:
            num = int(f[14:-5])
        elif i == 1:
            num = int(f[35:-5])
        else:
            num = int(f[36:-5])

        clus_num, m, q = para[num]
        result = result.append(pd.DataFrame([[clus_num, m, q] + list(metric.iloc[0])],
                                            columns=['clusters', 'm', 'q', 'precision', 'recall', 'F1']),
                               ignore_index=True)

    writer = pd.ExcelWriter('../../result/' + filename + '_para.xlsx')
    result.to_excel(writer, index=None, columns=['clusters', 'm', 'q', 'precision', 'recall', 'F1'])
    writer.save()
