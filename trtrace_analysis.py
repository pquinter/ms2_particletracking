from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_hmap(hmap, drop='movie', save=False, normtrace=True):
    try: hmap = hmap.drop(drop, axis=1)
    except ValueError: pass
    if normtrace:
        hmap = hmap.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=1)
    fig = plt.figure(figsize=(16, 10))
    sns.heatmap(hmap,xticklabels=5, yticklabels=False, cmap='viridis',
            robust=True)
    plt.xlabel('Time (s)')
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
nuclei_peaks = pd.read_pickle('../output/pp27/pp7spots_SVMfiltered.pkl')

# create mass heatmap
hmap = pd.DataFrame()
movcount = defaultdict(int)
for name, group in nuclei_peaks.groupby('imname'):
    h = tp.filter_stubs(group, 5)
    h = group[['mass','frame', 'particle']].pivot(index='particle', columns='frame')
    h['movie'] = name
    hmap = pd.concat((hmap, h), axis=0)
    movcount[name]+=h.shape[0]
# convert frame numbers to time in seconds, last column is movie name
hmap.columns = list(np.arange(20, 20*hmap.shape[1], 20)) + ['movie']
plot_hmap(hmap, normtrace=True)

# Align traces to the left
hmap_aligned = hmap.drop('movie', axis=1).apply(lambda x: align_trace(x,
        interpolate=np.mean), axis=1).dropna(how='all', axis=1)
plot_hmap(hmap_aligned)

# tidy dataframe for aligned scatter plot; it's a mess, not very useful
peaks_tidy = pd.melt(hmap_aligned)
peaks_tidy['movie'] = list(hmap['movie'])*hmap_aligned.shape[1]
peaks_tidy.columns = ['time', 'intensity', 'movie']
sns.tsplot(time='time', value='intensity', condition='movie', data=peaks_tidy,
            estimator=np.nanmean, ci=68, n_boot=1e2)
