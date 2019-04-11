import pandas as pd
import numpy as np
import glob

part_dir = '../output/pipeline_snapshots/particles'
parts = pd.read_csv(part_dir+'/parts_labeled.csv')
parts = parts[~(parts.strain.isin(['yQC25', 'TL47pQC116', 'TL47pQC99']))]
parts['strain'] = parts.strain.str.replace('yQC63TL174','yQC63')

# read segmentation data
seg_dir = '../output/pipeline_snapshots/segmentation'
seg_dir = glob.glob(seg_dir+'/*csv')
seg_df = []
for sdir in seg_dir:
    _df = pd.read_csv(sdir)
    _df['mov_name'] = sdir.split('/')[-1][:-4]
    seg_df.append(_df)
seg_df = pd.concat(seg_df, ignore_index=True)
# remove '_GFPlow' substring
seg_df['mov_name'] = seg_df.mov_name.apply(lambda x: x[:-9])

# get fraction of active cells per field of view per fluor thresh
freq_df = pd.DataFrame()
thresh_arr = np.arange(4,20)
for thresh in thresh_arr:
    filt_ix = (parts.mass_norm>thresh)&(parts.corrwideal>=0.5)
    part_count = parts[filt_ix].groupby(['time_postinduction','mov_name','strain']).count().x.reset_index()
    part_count.columns = ['time_postinduction','mov_name','strain','no_TS']
    cell_count = seg_df.groupby('mov_name').count().x_cell.reset_index()
    cell_count.columns = ['mov_name','no_cells']
    _freq_df = pd.merge(part_count, cell_count, on='mov_name')
    _freq_df['frac_active'] = _freq_df.no_TS / _freq_df.no_cells
    _freq_df['thresh'] = thresh
    freq_df = freq_df.append(_freq_df).reset_index(drop=True)
freq_df['rep'] = freq_df.mov_name.apply(lambda x: int(x.split('_')[-1]))
freq_df['rep'] = freq_df.rep.replace({72:7.5})

##### need to improve how to identify active TS
# filter by mov_name
#filt_bymov = []
#for im, df in parts.groupby('mov_name'):
#    filt_bymov.append(df[(df.mass_norm>df.mass_norm.mean())])
#filt_bymov = pd.concat(filt_bymov, ignore_index=True)
#part_count = filt_bymov.groupby(['time_postinduction','mov_name','strain']).count().x.reset_index()
#part_count.columns = ['time_postinduction','mov_name','strain','no_TS']
#cell_count = seg_df.groupby('mov_name').count().x_cell.reset_index()
#cell_count.columns = ['mov_name','no_cells']
#freq_df = pd.merge(part_count, cell_count, on='mov_name')
#freq_df['frac_active'] = freq_df.no_TS / freq_df.no_cells

plot_dir = '../output/pipeline_snapshots/plots/'
###############################################################################
# plot all replicates by threshold
###############################################################################
freq_thresh_hmap = freq_df.pivot_table(index=['thresh','rep'], columns='strain', values='frac_active')
fig, ax = plt.subplots(figsize=(5,8))
sns.heatmap(freq_thresh_hmap, ax=ax, cmap='viridis', robust=True, cbar_kws={'label':'Fraction active cells'})
plt.tight_layout()
plt.savefig(plot_dir+'hmap_rep_activecellfrac.pdf', bbox_inches='tight')

###############################################################################
# plot replicates in stripplot instead of heatmap
import matplotlib.cm as cm
fig, axes = plt.subplots(2,8, sharex=False, sharey=False, figsize=(22,6))
time_pal = cm.magma(np.linspace(0,0.8, len(parts.time_postinduction.unique())))
for (thresh, _freq_df), ax in zip(freq_df.groupby('thresh'), axes.ravel()):
    ax.tick_params(axis='x', labelsize=10, rotation=60)
    ax.tick_params(axis='y', labelsize=10, rotation=60)
    sns.stripplot(x='strain', y='frac_active', data=_freq_df, ax=ax,
            alpha=0.5, hue=_freq_df.time_postinduction, palette=time_pal)
    #sns.pointplot(x='strain', y='frac_active', data=_freq_df, ax=ax, alpha=0.5, size=5, join=False)
    ax.legend('', frameon=False)
    ax.annotate(' thresh={}'.format(thresh), (ax.get_xlim()[0], ax.get_ylim()[1]), fontsize=12)
[ax.set(ylabel='', xlabel='') for ax in axes.ravel()]
[ax.set_xticks([]) for ax in axes[0,:]]
# label and move to the middle for both axes rows
axes[0,0].set(ylabel='Active cells fraction')
axes[0,0].yaxis.set_label_coords(-0.5,0)
sns.despine()
plt.savefig(plot_dir+'strip_thresharr_activecellfrac_zoom_timecolor.pdf', bbox_inches='tight')

###############################################################################
# Plot heatmap of mean number of active cells by threshold
###############################################################################
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

###############################################################################
# plot most relevant single threshold
###############################################################################
thresh = 7
colors = cm.magma(np.linspace(0,0.8, len(freq_df['time_postinduction'].unique())))
fig, ax = plt.subplots(figsize=(5,8))
sns.stripplot(x='strain', y='frac_active', data=freq_df[freq_df.thresh==thresh],
        ax=ax, alpha=0.5, hue=freq_df[freq_df.thresh==thresh].time_postinduction, palette=colors, size=10)
sns.pointplot(x='strain', y='frac_active', data=freq_df[freq_df.thresh==thresh],
        ax=ax, alpha=0.5, color='#2f7585', size=10, join=False)
plt.legend([], frameon=False)
sns.despine()
ax.set(ylabel='Active cells fraction')
ax.tick_params(axis='x', labelsize=20, rotation=60)
plt.tight_layout()
plt.savefig(plot_dir+'strip_thresh{}massnorm_activecellfrac_timecolor.pdf'.format(thresh), bbox_inches='tight')
