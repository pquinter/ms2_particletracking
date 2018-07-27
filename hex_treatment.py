"""
hexanediol treatment seems to work, eliminates bright particles formed with FUS and brings them back to WT
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from skimage import io
from im_utils import *
import matplotlib.patches as mpatches

%matplotlib

hex75 = io.imread('../data/HexanediolTreatment/MAX_TL47pQC75_45minGalInduced_HexTreatFrame3.tif')
prehex75 = io.imread('../data/HexanediolTreatment/MAX_TL47pQC75_35minGalInduced_PreHexTreatment.tif')
hex92 = io.imread('../data/HexanediolTreatment/MAX_TL47tl092_30minGalInduced_HexTreatment.tif')
hex92_2 = io.imread('../data/HexanediolTreatment/MAX_TL47tl092_40minGalInduced_HexTreatment_2.tif')
hex92 = np.concatenate((hex92, hex92_2))
prehex92 = io.imread('/Users/porfirio/Documents/research/sternberg_lab/yeastEP/ms2pp7/data/HexanediolTreatment/MAX_TL47tl092_20minGalInduced_PreHexTreatment.tif')
def mean_med(mov):
    fmean = [np.mean(f) for f in mov]
    fmed = [np.median(f) for f in mov]
    return fmean, fmed
def plot(frame):
    fig, ax = plt.subplots()
    ax.imshow(frame)
    return ax

hmean, hmed = mean_med(hex75)
phmean, phmed = mean_med(prehex75)
fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
for ax, (mean, med) in zip(axes, (mean_med(hex75), mean_med(prehex75))):
    ax.plot(mean)
    ax.plot(med)
# bleaching happens at ~15 frames from exponential fit to mean

# Hexanediol treatment
roi = (slice(1836, 1903, None), slice(447, 523, None))
test = np.stack([f[roi] for f in hex75])
trace = [np.max(f) for f in test]

# pre Hexanediol treatment
roi = (slice(831, 909, None), slice(409, 478, None))
test_pre = np.stack([f[roi] for f in prehex75])
trace_pre = [np.max(f) for f in test_pre]

plt.plot(trace/max(trace), label='treatment')
plt.plot(trace_pre/max(trace_pre), label='pre')
plt.plot(trace_pre2/max(trace_pre2), label='pre2')
plt.axvline(3, c='b', ls='--', label='hex addition')
plt.axvline(15, c='k', ls='--', label='typ bleach time')
plt.legend()

ax = plot(hex75[0])
test_pre2 = np.stack([f[roi] for f in prehex75])
trace_pre2 = [np.max(f) for f in test_pre2]

def kymograph(mov, ax=None, t=20, vmin=100, vmax=750, ylabel=True, title=None, 
        xticks=0, log=False):
    """ Make and plot kymograph of movie """
    if ax is None:
        fig, ax = plt.subplots()
    # max intensity projection along columns; x coord is same in mov as in plot
    rowmax = pd.DataFrame([np.max(f, axis=0) for f in mov])
    # convert frames to minutes
    rowmax.index = [int(i) for i in rowmax.index*t/60]
    if log:
        rowmax = np.log(rowmax)
        vmin, vmax = np.min(rowmax.values), np.max(rowmax.values)
    sns.heatmap(rowmax, ax=ax, cmap='viridis', xticklabels=xticks,
        yticklabels=6, vmin=vmin, vmax=vmax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=10)
    if ylabel: ax.set_ylabel('Time (min)')
    if title: ax.set_title(title)
    return ax

labels = ('FUS-PP7-GFP', 'FUS-PP7-GFP +Hex', 'PP7-GFP', 'PP7-GFP +Hex')
fig, axes = plt.subplots(2, 2, sharey='row', figsize=(22, 8))
for i, (m, n) in zip((3,2,1,0), zip((prehex75, hex75, prehex92, hex92), labels)):
    ax = axes.flat[i]
    kymograph(m, ax, vmax=1500 if '75' in n else 750, ylabel=False, title=n)
axes[0,0].axhline(5, ls='--', alpha=0.4, c='w')
axes[1,0].axhline(3, ls='--', alpha=0.4, c='w')
axes[0,0].set_ylabel('Time (min)')
axes[1,0].set_ylabel('Time (min)')
plt.tight_layout()
plt.savefig('../output/hexTreatment/kymographs_75vWT.png', dpi= 70fig, axes = plt.subplots(2, sharex=True, sharey=True)

rowhex75 = np.array([np.max(f, axis=0) for f in hex75])
rowprehex75 = np.array([np.max(f, axis=0) for f in prehex75])
rowhex92 = np.array([np.max(f, axis=0) for f in hex92])

fig, ax = plt.subplots()
ax=axes[0]
color_ix = np.linspace(0, 1, len(rowhex75))
[plot_ecdf(f, ax=ax, alpha=0.1, color=plt.cm.plasma(i)) for f,i in zip(rowhex75, color_ix)]

ax=axes[1]
[plot_ecdf(f, ax=ax, alpha=0.1, color=plt.cm.plasma(i)) for f,i in zip(rowprehex75, color_ix)]

ax=axes[1]
color_ix = np.linspace(0, 1, len(rowhex92))
[plot_ecdf(f, ax=ax, alpha=0.1, color=plt.cm.plasma(i)) for f,i in zip(rowhex92, color_ix)]

# get top n maximums
n = 10
tidyhex = pd.DataFrame()
for arr, label in zip((prehex75, hex75, prehex92, hex92), labels):
    rowmax = np.array([np.max(f, axis=0) for f in arr])
    _df = pd.DataFrame([np.sort(f)[-n:] for f in rowmax]).T
    _df['label'] = label
    tidyhex = pd.concat((tidyhex, _df), axis=0)
tidyhex = pd.melt(tidyhex, id_vars='label')
tidyhex.columns = ('label', 'time', 'Fluor. (a.u.)')
# convert time to minutes
tidyhex['time'] = tidyhex['time'].apply(lambda x: x*20/60)


# make dicitonary of colors and legend patches
cmap=plt.cm.Dark2
colors = {l:cmap(c) for l, c in zip(labels, np.linspace(0, 1, len(labels)))}
patches = [mpatches.Patch(color=colors[l], label=l) for l in labels]


fig, ax = plt.subplots(1, figsize=(12,8))
sns.stripplot(x='time', y='Fluor. (a.u.)', data=tidyhex, hue='label', alpha=0.2, size=5, palette=colors)
sns.pointplot(x='time', y='Fluor. (a.u.)', data=tidyhex, hue='label', palette=colors)
plt.axvline(3, ls='--')

plt.legend(handles=patches)
xticks_pos = np.arange(0, len(tidyhex.time.unique()), 5)
xticks_val = np.linspace(0, tidyhex.time.max(), len(xticks_pos), dtype=int)
plt.xticks(xticks_pos, xticks_val, rotation=60)
ax.set(xlabel='Time (min)', ylabel='Fluor. (a.u.)')
sns.despine()
plt.tight_layout()
plt.savefig('../output/hexTreatment/topFoci_75hex.png', dpi= 300, bbox_inches='tight')

# take a look at 2 manually cropped cells
cprehex = io.imread('../data/HexanediolTreatment/croppedcell_MAX_TL47pQC75_35minGalInduced_PreHexTreatment.tif')
chex = io.imread('../data/HexanediolTreatment/croppedcell_MAX_TL47pQC75_45minGalInduced_HexTreatFrame3.tif')
cprehex = [np.max(cprehex[i,:,:]) for i in np.arange(0, len(cprehex))]
chex = [np.max(chex[i,:,:]) for i in np.arange(0, len(cprehex))]
plt.plot(cprehex, '.')
plt.plot(chex, '.')

pqc72 = io.imread('/Users/porfirio/Documents/research/sternberg_lab/yeastEP/ms2pp7/data/72_TL092/MAX_05142018_pQC725_TL092_4hPostInduction.tif')
pqc72_2 = io.imread('/Users/porfirio/Documents/research/sternberg_lab/yeastEP/ms2pp7/data/72_TL092/MAX_05142018_pQC725_TL092_3hPostInduction.tif')
kymograph(pqc72, log=True)
kymograph(pqc72_2, xticks=20, log=True)
