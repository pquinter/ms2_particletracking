from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle

with open('../output/pp7/05Feb2018sel_movs.p', 'rb') as f:
     sel_peaks = pickle.load(f)
#normalize by group
sel_peaks['norm_mass'] = sel_peaks[['cpid','mass']].groupby('cpid').transform(lambda x: (x - x.mean()) / x.std())
sel_peaks['norm_int'] = sel_peaks[['cpid','intensity']].groupby('cpid').transform(lambda x: (x - x.mean()) / x.std())
# smooth with sliding mean
wsize=20
sel_peaks['smooth_int'] =\
    sel_peaks[['cpid','norm_int']].groupby('cpid').transform(lambda x:\
        pd.concat((pd.Series(np.full(wsize-1, np.mean(x))), x),
            ignore_index=True).rolling(wsize).mean().dropna())
sel_peaks['smooth_int'] = sel_peaks[['cpid','smooth_int']].groupby('cpid').transform(lambda x: (x - x.mean()) / x.std())
fig, ax = plt.subplots(1)
#sel_peaks[sel_peaks.cpid=='3.0_77.0_11282017_664_9'].plot(x='frame',y='nmass', ax=ax)#, kind='scatter')
sel_peaks[sel_peaks.cpid=='3.0_77.0_11282017_664_9'].plot(x='frame',y='norm_int', ax=ax, alpha=0.2)
sel_peaks[sel_peaks.cpid=='3.0_77.0_11282017_664_9'].plot(x='frame',y='smooth_int', ax=ax)
# plot a sample
np.random.seed(42)
sample = sel_peaks.cpid.unique()[np.random.choice(np.arange(0,
                            len(sel_peaks.cpid.unique())), 10, replace=False)]
fig, axes = plt.subplots(5, 2)
for ax, group in zip(axes.ravel(),
        sel_peaks[sel_peaks.cpid.isin(sample)].groupby('cpid')):
    group[1].plot(x='frame',y='hmm', ax=ax)
    group[1].plot(x='frame',y='norm_int', ax=ax, alpha=0.2)
    ax.set_title(group[0])
#    [ax.axvline(x, ls='dashed') for x in group[1].smooth_int.agg(lambda x:\
#                                scipy.signal.argrelmax(x.values, order=10)[0])]
#    [ax.axvline(x, ls='dashed') for x in group[1].smooth_int.agg(lambda x:\
#                                scipy.signal.find_peaks_cwt(x.values, np.arange(4, 8)))]
    [ax.axvline(x, ls='dashed') for x in group[1].smooth_int.agg(lambda x:\
                                            np.where(np.diff(np.sign(x.values)))[0])]
    ax.axhline(0, alpha=0.2)
    ax.legend([])


spl_test = sel_peaks[sel_peaks.cpid=='3.0_77.0_11282017_664_9'].rmean.dropna().values
t = np.arange(1, len(spl_test)+1, dtype=float)

# test HMM fit instead of smoothing ========================================
sel_peaks = sel_peaks.sort_values(by=['cpid', 'frame']).reset_index(drop=True)
# concatenate (normalized) traces with padding to feed IDL HMM program
traces_concat = np.concatenate([np.concatenate((vals.dropna().values, np.full(10, 0))) \
                    for _, vals in sel_peaks.groupby('cpid').norm_int])
np.savetxt('/Users/porfirio/Desktop/HMM/testconcat.txt', traces_concat, fmt='%0.2f')
# load result of idealized trace
hmmconcat = np.loadtxt('/Users/porfirio/Desktop/HMM/testconcat_idelized_traces.txt')
# convert to boolean for convenience
hmmconcat = hmmconcat>np.mean(hmmconcat)
plt.plot(traces_concat, alpha=0.2)
plt.plot(hmmconcat)

# split into original groups
# get group sizes, then indices to split
group_sizes = sel_peaks.groupby('cpid').size().values+10
split_ind = np.cumsum(group_sizes)

# split initial group to verify split worked as expected
group_split = np.split(traces_concat, split_ind)
group_split = np.concatenate([g[:-10] for g in group_split])
# assert correct split
assert(np.isclose(1.0, np.corrcoef(group_split, sel_peaks.norm_int)[1][1]))

# now do idealized trace
hmm_split = np.split(hmmconcat, split_ind)[:-1]
# remove zero padding
hmm_split = np.concatenate([g[:-10] for g in hmm_split])
sel_peaks['hmm'] = hmm_split
# find peaks, very easy with derivative of hmm
peak_ind = scipy.signal.argrelmax(np.diff(hmmconcat))[0]
# get durations of ON and OFF states
# groupby generates a new group every time the value of key function changes
from itertools import groupby
on = [sum(1 for i in g) for k,g in groupby(hmmconcat) if k]
off = [sum(1 for i in g) for k,g in groupby(hmmconcat) if not k]

# =============================================================================

# smooth signal ===============================================================
import scipy
import scipy.signal
def fitsplines(x, s=50, returnt=False):
    t = np.arange(1, len(x)+1, dtype=float)
    spl = scipy.interpolate.UnivariateSpline(t, x, s=s)
    # spl is now a callable function
    x_spline = spl(t)
    if returnt:
        return x_spline, t
    else:
        return x_spline

# quick look at number of peaks identified by this approach
# smooth all traces with splines
sel_peaks['smooth_int'] = sel_peaks[['cpid','nint']].groupby('cpid').transform(fitsplines)
# find peaks and get total number
peakstest = sel_peaks[['cpid','smooth_int']].groupby('cpid').agg(\
                        lambda x: len(scipy.signal.argrelmax(x.values, order=10)[0])).reset_index()
# get mean by movie
peakstest['imname'] = peakstest.cpid.str.split('_', n=2, expand=True)[2]
print(peakstest.groupby('imname').smooth_int.apply(lambda x: np.mean(x)/len(x)))

# smoothing and peak detection in single example
test = sel_peaks[sel_peaks.cpid=='3.0_77.0_11282017_664_9'].norm_int.values
spl_test, t = fitsplines(test, returnt=True, s=50)

# find peak maxima
(points, ) = scipy.signal.argrelmax(spl_test, order=5)
xmax = t[points]
ymax = spl_test[points]
plt.plot(t, spl_test)
plt.plot(t, test, alpha=0.3, c='#85C1E9')
plt.xticks(np.arange(1, len(t)+1, 5), rotation=60)
plt.tight_layout()
[plt.scatter(x,y, c='#F878A3') for x,y in zip(xmax, ymax)]
[plt.axvline(x, alpha=0.2, ls='dashed') for x in xmax]

# find minima
# could use to threshold which peaks to keep, e.g.: xmax[ymax-ymin>0.03]
(points_min, ) = scipy.signal.argrelmin(Vs_spline, order=3)
xmin = ts_spline[points_min]
ymin = Vs_spline[points_min]
[plt.scatter(x,y, c='#C0392B') for x,y in zip(xmin, ymin)]
#==============================================================================
