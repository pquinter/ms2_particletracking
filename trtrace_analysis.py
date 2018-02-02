from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict

def plot_hmap(hmap, drop='movie', save=False, normtrace=True):
    try: hmap = hmap.drop(drop, axis=1)
    except ValueError: pass
    if normtrace:
        hmap = hmap.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=1)
    fig = plt.figure(figsize=(16, 10))
    sns.heatmap(hmap,xticklabels=5, yticklabels=False, cmap='viridis',
            robust=True)
    plt.xlabel('Time (min)')
    plt.ylabel('Trace')
    plt.xticks(rotation=60)
    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches='tight')

def align_trace(trace, interpolate=np.mean):
    """ Remove all nan values to the left of the trace"""
    i=0
    for value in trace.values:
        if np.isnan(value): i+=1
        else: break
    trace_aligned = np.concatenate((trace[i:], trace[:i]))

    def interpolate_func(t, interpolate):
        """ interpolate missing values"""
        for i, value in enumerate(t):
            if np.isnan(value):
                if np.isnan(t[i+1]): break
                try: trace_aligned[i] = interpolate((t[i-1], t[i+1]))
                except IndexError: break
        return t

    if interpolate: trace_aligned = interpolate_func(trace_aligned, interpolate)
    return trace_aligned

# load nuclei and particle tracking data
peaks = pd.read_csv('../output/pp7/peaks_complete.csv')
# pid is not unique id anymore because empty cells all have NaN
peaks['cpid'] = peaks.apply(lambda x: str(x.label)+str(x.particle)+'_'+x.imname, axis=1)
# peaks can be used directly on cpid, or better to just drop empty cells
peaks_wparts = peaks.dropna(subset=['x','y'])

# create intensity heatmap
hmap = peaks_wparts.sort_values('imname').pivot(index='cpid',
                        columns='frame', values='intensity')
# convert frame numbers to time in minutes
hmap.columns = list(np.round(np.arange(0, 20*hmap.shape[1], 20)/60))
plot_hmap(hmap, normtrace=True)

# Align traces to the left by removing NaNs in mass col in intensity col
peaks_al = pd.DataFrame()
for pid, group in peaks.groupby('pid'):
    nan_ind = group.mass.fillna(method='ffill').isnull()
    peaks_al = pd.concat((peaks_al, group[~nan_ind]), ignore_index=True)

hmap_aligned = peaks_al.sort_values('imname').pivot(index='pid',
                                    columns='frame', values='intensity')
hmap_aligned = hmap_aligned.apply(lambda x: align_trace(x, interpolate=False), axis=1)
plot_hmap(hmap_aligned)

# merged traces by movie, it's a mess and doesn't really make sense to do this
sns.tsplot(time='frame', value='intensity', condition='imname', data=peaks_al,
            estimator=np.nanmean, ci=68, n_boot=1e2)
