from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from im_utils import plot_ecdf

part_dir = '../output/pipeline_snapshots/particles'
parts = pd.read_csv(part_dir+'/parts_labeled.csv')
################################################################################
################################################################################
colors = ['b','k','r','seagreen','mediumorchid']
colors = {s:c for s,c in zip(parts.strain.unique(), colors)}
fig, ax = plt.subplots()
parts[parts.mass_norm>8].groupby('strain').apply(lambda x: plot_ecdf(x['mass'].values, color=colors[x.name], ax=ax, label=x.name))
plt.legend()
plt.close('all')


from skimage import io
import trackpy as tp
from im_utils import plot_ecdf
im_name = '03052019_yQC21_255u100%int480_150msExp_30-45minPosGal_3_w2GFPlow'
im_name = '03052019_yQC22_255u100%int480_150msExp_30-50minPosGal_3_w2GFPlow'
seg_im = io.imread('../output/pipeline_snapshots/segmentation/{}.tif'.format(im_name))
im = io.imread('/Users/porfirio/lab/yeastEP/ms2pp7/data/2019_pp7Snapshots/03052019_yQC22/{}.tif'.format(im_name))
fig, ax = plt.subplots()
tp.annotate(parts[(parts.mass_norm<=8)&(parts.mov_name==im_name)], np.log(im), ax=ax)
ax.imshow(seg_im>0, alpha=0.3)
for metric in ('mass','raw_mass','mass_norm'):
    plot_ecdf(parts[(parts.GPCprob>0.5)&(parts.mov_name==im_name)][metric], title=metric)

colors = cm.magma(np.linspace(0, 1, len(parts.time_postinduction.unique())))
parts['exp'] = parts.mov_name.apply(lambda x: int(x.split('_')[-2]))
parts['exp'] = parts.exp.replace({72:7.5})

# no correlation between time and TS intensity
fig, ax = plt.subplots()
#parts[(parts.mass_norm>8)].groupby('strain').apply(lambda x: plot_ecdf(x.mass_norm, color=colors[x.exp]))
#parts[(parts.mass_norm>8)].plot(y='raw_mass', x='exp', ax=ax, kind='hexbin', cmap='viridis')#, alpha=0.05)
parts[(parts.strain>'yQC22')&(parts.mass_norm>8)].plot(y='raw_mass', x='exp', ax=ax, kind='hexbin', cmap='viridis')#, alpha=0.05)

fig, axes = plt.subplots(3, figsize=(10,14), sharex=True)
for i, metric in enumerate(('raw_mass','mass','mass_norm')):
    ax=axes[i]
    for strain, sdf in parts[(parts.mass_norm>0)].groupby('strain'):
        data = sdf.groupby(['time_postinduction'])[metric].mean()
        sns.regplot(data.index, data.values, '.', color=colors[strain], label=strain, ax=ax)
        ax.set(ylabel='Mean {}'.format(metric), xlabel='')
plt.legend(fontsize=12)
ax.set(xlim=(25, 55), xlabel='Time after induction (min)')
sns.despine()
plt.tight_layout()
plt.savefig(plot_dir+'Rep_meanmass.pdf', bbox_inches='tight')



import glob
parts = parts[~(parts.strain=='yQC25')]
# read segmentation data
seg_dir = '../output/pipeline_snapshots/segmentation'
seg_dir = glob.glob(seg_dir+'/*csv')
seg_df = []
for sdir in seg_dir:
    _df = pd.read_csv(sdir)
    _df['mov_name'] = sdir.split('/')[-1][:-4]
    seg_df.append(_df)
seg_df = pd.concat(seg_df, ignore_index=True)
seg_df['mov_name'] = seg_df.mov_name.apply(lambda x: x[:-9])

# get fraction of active cells per field of view per fluor thresh
freq_df = pd.DataFrame()
thresh_arr = np.arange(4,20)
for thresh in thresh_arr:
    filt_ix = parts.mass_norm>thresh
    part_count = parts[filt_ix].groupby(['time_postinduction','mov_name','strain']).count().x.reset_index()
    part_count.columns = ['time_postinduction','mov_name','strain','no_TS']
    cell_count = seg_df.groupby('mov_name').count().x_cell.reset_index()
    cell_count.columns = ['mov_name','no_cells']
    _freq_df = pd.merge(part_count, cell_count, on='mov_name')
    _freq_df['frac_active'] = _freq_df.no_TS / _freq_df.no_cells
    _freq_df['thresh'] = thresh
    freq_df = freq_df.append(_freq_df).reset_index(drop=True)
#freq_df['exp'] = freq_df.mov_name.apply(lambda x: int(x.split('_')[-2]))
#freq_df['exp'] = freq_df.exp.replace({72:7.5})

# make replicate number column to show all data points in heatmap after pivot
#freq_df['rep'] = freq_df.groupby(['strain','thresh']).transform(lambda x: np.arange(len(x))).no_cells
#freq_df['rep'] = freq_df.groupby(['strain','thresh']).exp.transform(lambda x: x-x.min())
# Does fraction of active cells change over the 20 min?
# some correlation with time, less active cells...but conclusion is the same
fig, ax = plt.subplots(figsize=(10,6))
for strain, _freq_df in freq_df[freq_df.thresh==8].groupby('strain'):
    sns.regplot(x='time_postinduction', y='frac_active', data=_freq_df, ax=ax, color=colors[strain], label=strain)
ax.set(ylabel='Active cells fraction', xlabel='Time after induction (min)')
plt.legend()
sns.despine()
plt.tight_layout()
plt.savefig(plot_dir+'Rep_fracActive.pdf', bbox_inches='tight')

plot_dir = '../output/pipeline_snapshots/plots/'
freq_thresh_hmap = freq_df.pivot_table(index=['thresh','rep'], columns='strain', values='frac_active')
# plot all replicates by threshold
fig, ax = plt.subplots(figsize=(5,8))
sns.heatmap(freq_thresh_hmap, ax=ax, cmap='viridis', robust=True, cbar_kws={'label':'Fraction active cells'})
plt.savefig(plot_dir+'hmap_rep_activecellfrac.pdf', bbox_inches='tight')

# get number of cells per strain
hmap_nocells = freq_df.pivot_table(index=['thresh','rep'], columns='strain', values='no_cells').groupby('thresh').sum()
# only annotate top row (all are the same)
hmap_nocells = hmap_nocells.values.astype(int).astype(str)
hmap_nocells[0] = ['n='+val for val in hmap_nocells[0]]
hmap_nocells[1:] = ''

# plot mean fraction per threshold annotated with num cells per exp
fig, ax = plt.subplots(figsize=(7,10))
sns.heatmap(freq_thresh_hmap.groupby('thresh').mean(), ax=ax, cmap='viridis',
        annot=hmap_nocells, annot_kws={"size": 12}, fmt='s',
        robust=True, cbar_kws={'label':'Fraction active cells'})
ax.set(ylabel='TS fluorescence threshold')
ax.tick_params(axis='y', labelsize=20, rotation=0)
ax.tick_params(axis='x', labelsize=20, rotation=60)
plt.tight_layout()
plt.savefig(plot_dir+'hmap_mean_activecellfrac.pdf', bbox_inches='tight')

# plot replicates in stripplot instead of heatmap
import matplotlib.cm as cm
fig, axes = plt.subplots(2,8, sharex=True, sharey=False, figsize=(22,6))
time_pal = cm.magma(np.linspace(0,0.8, len(parts.time_postinduction.unique())))
for (thresh, _freq_df), ax, c in zip(freq_df.groupby('thresh'), axes.ravel(), colors):
    ax.tick_params(axis='x', labelsize=10, rotation=60)
    ax.tick_params(axis='y', labelsize=10, rotation=60)
    sns.stripplot(x='strain', y='frac_active', data=_freq_df, ax=ax, alpha=0.5, hue=_freq_df.time_postinduction, palette=time_pal)
    #sns.pointplot(x='strain', y='frac_active', data=_freq_df, ax=ax, alpha=0.5, size=5, join=False)
    ax.legend('', frameon=False)
    ax.annotate(' thresh={}'.format(thresh), (ax.get_xlim()[0], ax.get_ylim()[1]), fontsize=12)
[ax.set(ylabel='', xlabel='') for ax in axes.ravel()]
# label and move to the middle for both axes rows
axes[0,0].set(ylabel='Active cells fraction')
axes[0,0].yaxis.set_label_coords(-0.5,0)
sns.despine()
plt.savefig(plot_dir+'strip_thresharr_activecellfrac_zoom_timecolor.pdf', bbox_inches='tight')

# plot most relevant single threshold
colors = cm.magma(np.linspace(0,0.8, len(freq_df['exp'].unique())))
fig, ax = plt.subplots(figsize=(5,8))
sns.stripplot(x='strain', y='frac_active', data=freq_df[freq_df.thresh==8],
        ax=ax, alpha=0.5, hue=freq_df[freq_df.thresh==8].exp, palette=colors, size=10)
sns.pointplot(x='strain', y='frac_active', data=freq_df[freq_df.thresh==8],
        ax=ax, alpha=0.5, color='#2f7585', size=10, join=False)
plt.legend([])
sns.despine()
ax.set(ylabel='Active cells fraction')
ax.tick_params(axis='x', labelsize=20, rotation=60)
plt.tight_layout()
plt.savefig(plot_dir+'strip_thresh8massnorm_activecellfrac_timecolor.pdf', bbox_inches='tight')
