#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn import metrics

result = pd.DataFrame()
for f in os.listdir('../../result/GutenTAG/poly-channels-single-of-20/'):
    print(f)
    scores = pd.read_excel('../../result/GutenTAG/poly-channels-single-of-20/' + f, sheet_name='scores', index_col=0)
    gt = pd.read_excel('../../result/GutenTAG/poly-channels-single-of-20/' + f, sheet_name='gt_label', index_col=0)
    num = int(f[36:-5])

    precision_ed, recall_ed, _ = metrics.precision_recall_curve(gt, scores['ed_scores'])
    precision_wed, recall_wed, _ = metrics.precision_recall_curve(gt, scores['wed_scores'])
    precision_wdtw, recall_wdtw, _ = metrics.precision_recall_curve(gt, scores['wdtw_scores'])

    metr = {'dataset': ['poly20'] * 3,
            'number': [num] * 3,
            'indx': ['ed', 'wed', 'wdtw'],
            'roc_auc': [metrics.roc_auc_score(gt, scores['ed_scores']),
                        metrics.roc_auc_score(gt, scores['wed_scores']),
                        metrics.roc_auc_score(gt, scores['wdtw_scores'])],
            'pr_auc': [metrics.auc(recall_ed, precision_ed),
                       metrics.auc(recall_wed, precision_wed),
                       metrics.auc(recall_wdtw, precision_wdtw)],
            'ap': [metrics.average_precision_score(gt, scores['ed_scores']),
                   metrics.average_precision_score(gt, scores['wed_scores']),
                   metrics.average_precision_score(gt, scores['wdtw_scores'])]
            }
    result = result.append(pd.DataFrame(metr), ignore_index=True)

writer = pd.ExcelWriter('../../result/poly20_metric.xlsx')
result.to_excel(writer, index=None)
writer.save()
