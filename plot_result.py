import os
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from collections import defaultdict
from tesseract.loader import load_features
from tesseract.temporal import time_aware_train_test_split
from tesseract import viz

line_kwargs = {'linewidth': 1, 'markersize': 5}


# input the full path of ICE/CCE/TCE result from dir '..../timeseries_cred_conf/XXX_results.p'
def plot(result_path):
    results = pickle.load(open(result_path, 'rb'))
    data_tables = defaultdict(list)

    for j in range(len(results)):
        data_tables['f1_b'].append(results[j]['f1_b'])
        data_tables['f1_r'].append(results[j]['f1_r'])
        data_tables['f1_k'].append(results[j]['f1_k'])
        data_tables['reject_total_perc'].append(results[j]['reject_total_perc'])
        data_tables['reject_neg_perc'].append(results[j]['reject_neg_perc'])
        data_tables['reject_pos_perc'].append(results[j]['reject_pos_perc'])
    results = pd.DataFrame.from_dict(data_tables)

    viz.set_style()
    fig, (ax1) = plt.subplots(1, 1)

    # plot total reject bar
    ax1.bar(results.index, results['reject_total_perc'], width=0.7, color='#C0C0C0', alpha=0.6)

    # plot o (malware drift rate)
    ax1.scatter(results.index, results['reject_pos_perc'], facecolors='none', edgecolors='gray', marker='o',
                alpha=1.0, label='Rate of drifting malware')
    # plot x (goodware drift rate)
    ax1.scatter(results.index, results['reject_neg_perc'], c='gray', marker='x',
                alpha=1.0, label='Rate of drifting goodware')

    # plot kept F1
    ax1.plot(results.index, results['f1_k'], label='F1 (keep)', alpha=1.0, linestyle='-',
             marker='o', c='dodgerblue', markeredgewidth=1, **line_kwargs)

    # plot baseline (full dataset)
    ax1.plot(results.index, results['f1_b'], label='F1 (no rejection)', alpha=1.0, linestyle='--',
             c='gray', markeredgewidth=1, **line_kwargs)

    # plot reject F1
    ax1.plot(results.index, results['f1_r'], label='F1 (rejection)', alpha=1.0, marker='o',
             c='red', markeredgewidth=1, **line_kwargs)

    viz.set_title_sc(ax1, result_path.split('/')[-1])
    add_legend(ax1)

    viz.style_axes((ax1,), len(results.index))
    fig.set_size_inches(6, 4)
    plt.tight_layout()

    viz.save_images(plt, result_path.split('/')[-1])

    print('Plot ready!')


def add_legend(ax, loc='lower left'):
    # lines = ax.get_lines()
    legend = ax.legend(frameon=True, loc=loc, prop={'size': 10})
    legend.get_frame().set_facecolor('#FFFFFF')
    legend.get_frame().set_linewidth(0)
    return legend
