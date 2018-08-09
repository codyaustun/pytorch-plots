import numpy as np
from matplotlib import pyplot as plt

from resnet.cifar10.results import *

def plot_accuracy_over_time(df, label=None, title=None, y=None, x=None, filename=None, ax=None, style='-', **kwargs):
    yticks = np.linspace(0.5, 1, 11)
    y = y or 'Top-1 Accuracy'
    x = x or 'Time (in hours)'
    
    if ax is None:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)

    print("{} max acc ({}) in {}".format(label, df['accuracy'].max(), df['elapsed'].iloc[-1]))
    ax.plot(df['elapsed'].astype('timedelta64[s]') / 3600, df['accuracy'], label=label, style='-', **kwargs)

    ax.set_title(title)
    ax.set_ylabel(y)
    ax.set_yticks(yticks)
    ax.set_xlabel(x)
    ax.grid(which='both')
    ax.legend(loc=4)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    return ax

def plot_accuracy_over_epoch(df, label=None, title=None, y=None, x=None, filename=None, ax=None, style='-', **kwargs):
    yticks = np.linspace(0.5, 1, 11)
    y = y or 'Top-1 Accuracy'
    x = x or 'Epochs'
    
    if ax is None:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        
    data = df.groupby('epoch')['accuracy'].agg(['mean', 'std'])
    max_accs = df.groupby('run')['accuracy'].agg('max')

    print("{} mean ({:.6f}) - std ({:.6f}) - max acc ({}) - min acc ({})"
          .format(label, 
                  max_accs.mean(),
                  max_accs.std(),
                  max_accs.max(),
                  max_accs.min()))
    ax.plot(data.index, data['mean'], label=label, linestyle=style, **kwargs)
    ax.fill_between(data.index, data['mean'] + data['std'], data['mean'] - data['std'],
                    alpha=0.5, **kwargs)

    ax.set_title(title)
    ax.set_ylabel(y)
    ax.set_yticks(yticks)
    ax.set_xlabel(x)
    ax.grid(which='both')
    ax.legend(loc=4)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    return ax

def time_to_threshold(df, threshold):
    best = df[df['accuracy'] >= threshold]
    if len(best):
        record = {
            'elapsed': best.iloc[0, :]['elapsed'],
            'epoch': best.iloc[0, :]['epoch'],
            'accuracy': best.iloc[0, :]['accuracy'],
        }
    else: 
        record = None
    return record

import math
import itertools

def plot_grid(rows, cols, data, x, y, zscore=True, window=None, **kwargs):
    nplots = max(len(x), len(y))
    x = itertools.cycle(x)
    y = itertools.cycle(y)
    assert not (rows == -1 and cols == -1), "rows or cols needs to not be -1"
    rows = rows if rows != -1 else math.ceil(nplots / cols)
    cols = cols if cols != -1 else math.ceil(nplots / rows)
    
    fig = plt.figure(**kwargs)
    for index, (_x, _y) in enumerate(zip(x, y)):
        ax = fig.add_subplot(rows, cols, index + 1)
        x_values = data[_x]
        y_values = data[_y]
        title = _y
        xlabel = _x
        ylabel = _y
        if zscore:
            mean = y_values.mean()
            std = y_values.std()
            y_values = (y_values - mean) / std
            ylabel += ' zscore'
            title += " mean {:.4E} and std {:.4E} ".format(mean, std)
            
        if window is not None:
            y_values = y_values.rolling(window).mean()
            
        ax.plot(x_values, y_values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if index + 1 == nplots:
            break

def get_finished(test, name):
    latest = sorted(test['run'].unique())[-1]
    finished = test[test['epoch'] == test['epoch'].max()]['run'].unique()
    unfinished = test[~test['run'].isin(finished)]['run'].unique()

    print('{} Finished runs for {}'.format(len(finished), name))
    print('{} Unfinished runs for {}'.format(len(unfinished), name))
    return latest, finished, unfinished


def get_subset_records(configs, test, name, rank_metric='margin', desc=False, **kwargs):
    sizes = {c['subset'] for c in configs.values()}
    latest, finished, unfinished = get_finished(test, name)
    
    records = []
    for size in sorted(sizes):
        criteria = {'subset': size, 'rank_metric': rank_metric, 'desc': desc}
        criteria.update(kwargs)
        # print(criteria)

        runs = search_configs(criteria, configs)
        df = test[test['run'].isin(runs) & test['run'].isin(finished)]
        max_accs = df.groupby('run')['accuracy'].max()
        records.append({
            'nruns': len(max_accs),
            'size': size,
            'max': max_accs.max(),
            'min': max_accs.min(),
            'mean': max_accs.mean(),
            'std': max_accs.std(),
        })
    return records


