from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

with open('../output/pp7/05Feb2018sel_movs.p', 'rb') as f:
     sel_peaks = pickle.load(f)
#normalize by group
normed = pd.DataFrame()
mass, inten, cpid = [], [], []
for name, group in sel_peaks.groupby('cpid'):
    mass.append(group['mass']/group['mass'].max())
    inten.append(group['intensity']/group['intensity'].max())
    inten.append(group['intensity']/group['intensity'].max())

normed_mass = sel_peaks[['cpid','mass']].groupby('cpid').transform(lambda x: (x - x.mean()) / x.std())
normed_int = sel_peaks[['cpid','intensity']].groupby('cpid').transform(lambda x: (x - x.mean()) / x.std())
sel_peaks['nmass'] = normed_mass
sel_peaks['nint']= normed_int
fig, ax = plt.subplots(1)
sel_peaks[sel_peaks.cpid=='3.0_77.0_11282017_664_9'].plot(x='frame',y='nmass', ax=ax)#, kind='scatter')
sel_peaks[sel_peaks.cpid=='3.0_77.0_11282017_664_9'].plot(x='frame',y='nint', ax=ax, kind='scatter')

# smooth signal ===============================================================
import scipy

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
# smooth all traces
sel_peaks['smooth_int'] = sel_peaks[['cpid','nint']].groupby('cpid').transform(fitsplines)
# find peaks and get total number
peakstest = sel_peaks[['cpid','smooth_int']].groupby('cpid').agg(\
                        lambda x: len(scipy.signal.argrelmax(x.values)[0])).reset_index()
# get median by movie
peakstest['imname'] = peakstest.cpid.str.split('_', n=2, expand=True)[2]
print(peakstest.groupby('imname').smooth_int.mean())

# smoothing and peak detection in single example
test = sel_peaks[sel_peaks.cpid=='3.0_77.0_11282017_664_9'].nint.values
spl_test, t = fitsplines(test, returnt=True, s=50)
# find peak maxima
(points, ) = scipy.signal.argrelmax(spl_test)
xmax = t[points]
ymax = spl_test[points]
plt.clf()
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
