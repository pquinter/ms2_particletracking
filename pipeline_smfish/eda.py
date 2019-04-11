from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import corner
from utils import particle
from im_utils import *
import matplotlib.patches as mpatches
import re

part_dir = '../output/pipeline_smfish/particles/parts_filtered.csv'
parts = pd.read_csv(part_dir)
parts['logmass'] = np.log(parts.raw_mass.values)
plt.close('all')
for s, _parts in parts.groupby('strain'):
    fig = corner.corner(_parts[['corrwideal','logmass']])
    axes = fig.get_axes()
    axes[0].set_title(s)

parts['cid'] = parts['cell'].apply(str) + parts['mov_name']
parts['date'] = parts.mov_name.apply(lambda x: re.match('.*(\d{8}).*', x).group(1))
count = parts[parts.corrwideal>0.3].groupby(['cid','strain']).count().x.reset_index()

# get top particles per cell
value = 'mass'
max_parts = parts[(parts.nucleus>0)&(parts.corrwideal>=0.5)&(parts[value]>parts[value].mean()*1.2)].sort_values(value, ascending=False).groupby('cid').first().reset_index()

# add CTDr col for plot
rep_dict = {strain:r for strain, r in zip(parts.strain.unique(), (11,10,12))}
max_parts['CTDr'] = max_parts.strain.map(rep_dict)
multi_ecdf(max_parts, value=value, alpha=0.2, stack=False, plot_pooled=True, groupby=['CTDr','mov_name'], save=outdir+'bpmass_TSonly_byimage_pooled.pdf', legend=False)
plt.legend([])


# look at dispersion
mean_var = max_parts.groupby(['strain','mov_name']).apply(lambda x: (x[value].var(),x[value].mean(), x[value].median())).apply(pd.Series).reset_index()
mean_var.columns = ['strain','mov_name','var','mean','median']
mean_var['var/mean'] = mean_var['var'] / mean_var['mean']

outdir = '../output/pipeline_smfish/plots/'
plot_value = 'var/mean'
with sns.axes_style(*style):
    fig, ax = plt.subplots(figsize=(7,10))
sns.stripplot(x='strain', y=plot_value, data=mean_var, ax=ax, alpha=0.5, s=10, color='#006080')
sns.pointplot(x='strain', y=plot_value, data=mean_var, ax=ax, alpha=0.5, s=10, color='#966674', join=False, ci=99)
ax.set(ylabel='TS {} fluorescence (a.u.)'.format(plot_value))
ax.set(ylabel='TS variance/mean')
sns.despine()
plt.tight_layout()
ax.set_xticklabels(['FUS','TAF','-'], rotation=60)
plt.savefig(outdir+'varmean'+'.pdf', bbox_inches='tight')



fig, ax = plt.subplots()
parts[(parts.pid.isin(max_parts))].groupby('strain').apply(lambda x: plot_ecdf(x.raw_mass.values, ax=ax, color=None, label=x.name))
plt.legend()

cell_fluor = parts[parts.corrwideal>=0.3].groupby(['strain','cid']).mass.sum().reset_index()

fig, ax = plt.subplots()
cell_fluor.groupby('strain').apply(lambda x: plot_ecdf(x.mass.values, color=None, ax=ax, label=x.name))
plt.legend()

plt.close('all')
spots_dir = '../output/pipeline_smfish/spot_images'
pids_all, rawims_all, bpims_all = particle.load_patches(spots_dir)

goodparts = set(parts[parts.corrwideal>=0.3].pid.values)
pid_filt = [pid in goodparts for pid in pids_all]
pid_filtwt = np.array(['tl092' in pid for pid in pids_all]) & np.array(pid_filt)
pid_filt75 = np.array(['pQC75' in pid for pid in pids_all]) & np.array(pid_filt)
pid_filt76 = np.array(['pQC76' in pid for pid in pids_all]) & np.array(pid_filt)

vmin, vmax = rawims_all.min(), rawims_all.max()

fig, axes = plt.subplots(3)
for filt_ix, ax in zip((pid_filtwt, pid_filt75, pid_filt76), axes.ravel()):
    ax.imshow(im_block(rawims_all[filt_ix], cols=200, norm=True, sort=corr_widealspot), vmin=0, vmax=1, cmap='viridis')

import trackpy as tp
impath = '/Users/porfirio/lab/yeastEP/smFISH/data/TL47pQC7576/05052018/TL47pQC75/05052018_TL47pQC75_5.tif'
impath = '/Users/porfirio/lab/yeastEP/smFISH/data/TL47pQC7576/04242018/TL47tl092/04242018_TL47tl092_11.tif'
impath = '/Users/porfirio/lab/yeastEP/smFISH/data/TL47pQC7576/04242018/TL47pQC76/04242018_TL47pQC76_9.tif'
im = io.imread(impath)[:,:,0]
imname = impath.split('/')[-1][:-4]
plt.figure()
tp.annotate(max_parts[(max_parts.mov_name==imname)&(max_parts.nucleus>=0)], np.log(im))
