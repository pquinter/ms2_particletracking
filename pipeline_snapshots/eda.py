from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from im_utils import plot_ecdf
from matplotlib import cm
import matplotlib.patches as mpatches

part_dir = '../output/pipeline_snapshots/particles'
parts = pd.read_csv(part_dir+'/parts_filtered.csv')

parts = pd.read_csv('../output/pipeline_snapshots/particles/parts_filtered_10partsPerCell.csv')
parts = parts[~(parts.strain=='yQC25')]
rep_dict = {'yQC21':15, 'yQC22':14,'yQC23':12,'yQC62':10, 'yQC26':10.1,
        'yQC63TL174':9, 'TL47pQC99':33, 'TL47pQC116':30 , 'TL47pQC115':32, 'TL47pQC118':31,
        'TL47pQC119S2':27, 'TL47pQC1195D':28, 'TL47pQC1203K':27.1, 'TL47pQC1202H':26, 'TL47pQC1192E':29}
parts['CTDr'] = parts.strain.map(rep_dict)
parts['cid'] = parts['roi'].apply(str) + parts['mov_name']
################################################################################
################################################################################
plot_dir = '../output/pipeline_snapshots/plots/'
colors = ['b','k','r','seagreen','mediumorchid', 'g','y','cyan']
colors = cm.tab20c(np.linspace(0.1,1, len(parts.strain.unique())))
colors_strain = {s:c for s,c in zip(parts.strain.unique(), colors)}
patches = [mpatches.Patch(color=colors_strain[l], label=l) for l in colors_strain]
# filter by mov_name
filt_bymov = []
for im, df in parts.groupby('mov_name'):
    filt_bymov.append(df[(df.mass_norm>df.mass_norm.mean())])
filt_bymov = pd.concat(filt_bymov, ignore_index=True)
fig, ax = plt.subplots()
filt_bymov.groupby('strain').apply(lambda x: plot_ecdf(x['mass_norm'].values, color=colors[x.name], ax=ax, label=x.name))
plt.legend()

fig, ax = plt.subplots(figsize=(10,8))
parts[(parts.mass_norm>=7)&(parts.corrwideal>=0.5)].groupby('strain').apply(lambda x: plot_ecdf(x['mass_norm'].values,
                            color=colors_strain[x.name], ax=ax, label=x.name, formal=True, rasterized=True))
parts[(parts.mass_norm>=7)&(parts.corrwideal>=0.5)].groupby('strain').apply(lambda x: ax.axvline(np.median(x['mass_norm'].values),
    color=colors_strain[x.name], ymax=0.1, linewidth=3))
plt.ylim(0,1.05)
ax.annotate(r'$\leftarrow$median', (12, 0.05))
plt.legend(handles=patches)
plt.tight_layout()
plt.savefig(plot_dir+'ecdfburstsize_thresh7massnorma05corr.pdf', bbox_inches='tight')
#plt.close('all')
style = ("whitegrid", { 'grid.linestyle': '--'})
style= ("whitegrid", { 'grid.linestyle': '--', 'grid.color': '0.9', 'axes.edgecolor': '0.3', 'xtick.color': '0.3', 'ytick.color': '0.3', 'axes.labelcolor': '.3'})
def multi_ecdf(df, strains='all', value='mass_norm', alpha=0.8, plot_median=True,
        plot_pooled=False, cmap=cm.magma_r, save=False, xlim='auto', groupby=['CTDr', 'mov_name'],
        xlabel='Burst size', stack=True, legend=True):

    if not strains == 'all': df = df[(df.strain.isin(strains))]
    # get colors
    CTDr = sorted(df.CTDr.unique())
    ctd_labels = dict(df[['CTDr','strain']].values)
    try: colors_r = cmap(np.linspace(0.1,1, len(CTDr)))
    except TypeError: colors_r = cmap
    if isinstance(cmap, str): cmap = [cmap]*len(CTDr)
    colors_ctd = {r:c for r,c in zip(CTDr, colors_r)}
    patches_ctd = [mpatches.Patch(color=colors_ctd[l], label=ctd_labels[l]) for l in colors_ctd]

    with sns.axes_style(*style):
        fig, axes = plt.subplots(len(df.strain.unique()), sharex=True, sharey=True, figsize=(10,8))
    axes_dict = {s:ax for s,ax in zip(CTDr, axes.ravel())}
    grouped = df.groupby(groupby)
    grouped.apply(lambda x: plot_ecdf(x[value].values, alpha=alpha,
        color=colors_ctd[x.name[0]], ax=axes_dict[x.name[0]],
        label=x.strain.iloc[0], formal=1, ylabel=''))

    if plot_pooled:
        grouped = df.groupby('CTDr')
        grouped.apply(lambda x: plot_ecdf(x[value].values, alpha=1, lw=2,
            color=colors_ctd[x.name], ax=axes_dict[x.name],
            label=x.strain.iloc[0], formal=1, ylabel=''))
    if plot_median and stack:
        # plot median
        ys = np.linspace(-0.05, -0.05*len(CTDr), len(CTDr))
        yvals = {ctd:y for ctd,y in zip(CTDr, ys)}
        ## by replicate
        df.groupby(groupby).apply(lambda x:\
                axes[-1].scatter(np.median(x[value].values), yvals[x.name[0]],
                                s=50, alpha=0.2, color=colors_ctd[x.name[0]]))
        # get medians by group
        group_med = df.groupby(groupby).apply(lambda x:\
                (np.median(x[value].values), yvals[x.name[0]])).reset_index()
        group_med[['med','y']] = group_med[0].apply(pd.Series)
        # plot medians by group
        group_med.apply(lambda x: axes[-1].scatter(x.med, x.y, s=51, alpha=0.2,
            color=colors_ctd[x.CTDr]),axis=1)

        # plot median of medians
        group_med.groupby(groupby[0]).apply(lambda x:\
                axes[-1].scatter(np.median(x.med), np.median(x.y), s=120, alpha=0.6,
            color=colors_ctd[x.CTDr.iloc[0]]))

        ## plot median of everything pooled
        #df.groupby(groupby[0]).apply(lambda x:\
        #        axes[-1].scatter(np.median(x[value].values), yvals[x.name],
        #                        s=120, alpha=0.6, color=colors_ctd[x.name]))
        # plot 0, 0.5 and 1.0 ticks for each
    if stack:
        t_thick = {0:1, 0.5:1.5, 1:2}
        [axes_dict[ctd].axhline(t, -0.1, 0.015, lw=t_thick[t], c=colors_ctd[ctd])
                for ctd in axes_dict for t in (0, 0.5, 1)]
        # bring plots closer
        fig.subplots_adjust(hspace=-0.95)
        # make background transparent
        [ax.patch.set_visible(False) for ax in axes]
        # keep ticks for grid but hide repeated tick labels
        [ax.set_yticks([0, 0.5, 1]) for ax in axes]
        [ax.tick_params(axis='y', labelleft=False) for ax in axes]

        # replace above with below to do tick labels for each plot
        #[axes_dict[ctd].tick_params(axis='y', labelsize=10, colors=colors_ctd[ctd]) for ctd in axes_dict]
        # keep ytick labels only in middle plot
        axes[len(axes)//2].tick_params(axis='y', labelleft=True)
        # hide X axis
        [ax.xaxis.grid(False) for ax in axes]
        axes[-1].xaxis.grid(True)
        #[ax.set_xticks([]) for ax in axes]
        [ax.spines['bottom'].set_visible(False) for ax in axes[:-1]]
        #[ax.spines['left'].set_visible(False) for ax in axes[:-1]]
        # or uncomment to replace it with custom lines
        #[ax.axhline(0, c=c, ls='--', alpha=0.3) for ax,c in zip(axes, colors_r)]

    # make space for ylabel
    plt.subplots_adjust(left=0.11)
    # label y axis in middle plot
    axes[len(axes)//2].set(ylabel='ECDF')
    # label xaxis in last plot
    axes[-1].set(xlabel=xlabel)
    if legend:
        plt.legend(handles=patches_ctd, title='CTD repeats', loc=4)
    if xlim=='auto': xlim = (df[value].min()*0.1, df[value].max())
    if plot_median and stack:
        axes[-1].set(xlim=xlim, ylim=(min(ys)-0.05,1.05))
    else:
        axes[-1].set(xlim=xlim, ylim=(-0.05,1.05))
    if save:
        plt.savefig(save, bbox_inches='tight')
    return fig, axes

strains2plot = ['yQC21','TL47pQC99', 'TL47pQC116', 'TL47pQC115']
strains2plot = ['yQC21','yQC62', 'yQC26']
strains2plot = ['yQC21','yQC22', 'yQC23', 'yQC62', 'yQC63TL174']
strains2plot = ['yQC21','TL47pQC116','TL47pQC99','TL47pQC1192E','TL47pQC119S2','TL47pQC1195D']
strains2plot = ['yQC21','TL47pQC99', 'TL47pQC116','TL47pQC119S2', 'TL47pQC1195D', 'TL47pQC1192E', 'TL47pQC115', 'TL47pQC1203K','TL47pQC1202H']
strains2plot = ['yQC21','TL47pQC115', 'TL47pQC1203K','TL47pQC1202H', 'TL47pQC116']
colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0']
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
fig, axes = multi_ecdf(parts[(parts.mass_norm>=7)&(parts.corrwideal>=0.5)], strains2plot, value='mass_norm', plot_median=True, plot_pooled=True, alpha=0.3, stack=True)
            save=plot_dir+'burstsize/ecdf_size_sCTD.pdf')

# single plot
fig, ax = plt.subplots()
grouped = parts[(parts.strain.isin(strains2plot))&(parts.mass_norm>=8)&(parts.corrwideal>=0.5)].groupby(['strain', 'mov_name'])
grouped.apply(lambda x: plot_ecdf(x.mass_norm.values, alpha=0.5,
                                color=colors_strain[x.name[0]], ax=ax,
                                label=x.strain.iloc[0], formal=1, ylabel=''))
ax.set(ylabel='ECDF', xlabel='Burst size')
plt.tight_layout()
patches = [mpatches.Patch(color=colors_strain[l], label=l) for l in colors_strain if l in strains2plot]
plt.legend(handles=patches, loc=4)
plt.savefig(plot_dir+'ECDF_byStrain.pdf', bbox_inches='tight')



# plot nuclear fluorescence over time

ind_times = parts.time_postinduction.unique()
colors_time = cm.magma(np.linspace(0,0.8, len(ind_times)))
colors_time = {t:c for t,c in zip(sorted(ind_times), colors_time)}

fig, ax = plt.subplots()
parts[parts.strain.isin(strains2plot)].drop_duplicates(['mov_name','roi']).groupby(['strain', 'time_postinduction']).apply(lambda x: ax.scatter(x.name[1], x.nuc_fluor.mean(), color=colors_strain[x.name[0]], label=x.name[0]))
plt.legend(handles=patches)

fig, axes = plt.subplots(2, 3)
axes_dict = {s:ax for s,ax in zip(parts.strain.unique(), axes.ravel())}
parts.drop_duplicates(['mov_name','roi']).groupby(['strain', 'time_postinduction']).apply(lambda x: plot_ecdf(x['nuc_fluor'].values, color=colors_time[x.name[1]], ax=axes_dict[x.name[0]], label=x.name[0], formal=1, ylabel='', title=x.name[0]))
plt.tight_layout()

from skimage import io
import trackpy as tp
from im_utils import plot_ecdf
im_name = '03052019_yQC21_255u100%int480_150msExp_30-45minPosGal_3_w2GFPlow'
im_name = '03052019_yQC22_255u100%int480_150msExp_30-50minPosGal_3_w2GFPlow'
seg_im = io.imread('../output/pipeline_snapshots/segmentation/{}.tif'.format(im_name))
im = io.imread('/Users/porfirio/lab/yeastEP/ms2pp7/data/2019_pp7Snapshots/03052019_yQC22/{}.tif'.format(im_name))
fig, ax = plt.subplots()
tp.annotate(parts[(parts.mass_norm<=7)&(parts.mov_name==im_name)], np.log(im), ax=ax)
ax.imshow(seg_im>0, alpha=0.3)


colors = cm.magma(np.linspace(0, 1, len(parts.time_postinduction.unique())))
parts['exp'] = parts.mov_name.apply(lambda x: int(x.split('_')[-1]))
parts['exp'] = parts.exp.replace({72:7.5})

# no correlation between time and TS intensity
fig, ax = plt.subplots()
#parts[(parts.mass_norm>8)].groupby('strain').apply(lambda x: plot_ecdf(x.mass_norm, color=colors[x.exp]))
#parts[(parts.mass_norm>8)].plot(y='raw_mass', x='exp', ax=ax, kind='hexbin', cmap='viridis')#, alpha=0.05)
parts[(parts.strain>'yQC22')&(parts.mass_norm>8)].plot(y='raw_mass', x='time_postinduction', ax=ax, kind='hexbin', cmap='viridis')#, alpha=0.05)

fig, axes = plt.subplots(3, figsize=(10,14), sharex=True)
for i, metric in enumerate(('raw_mass','mass','mass_norm')):
    ax=axes[i]
    for strain, sdf in parts[(parts.strain.isin(strains2plot))&(parts.mass_norm>7)&(parts.corrwideal>=0.5)].groupby('strain'):
        data = sdf.groupby(['time_postinduction'])[metric].mean()
        sns.regplot(data.index, data.values, '.', color=colors_strain[strain], label=strain, ax=ax)
        ax.set(ylabel='Mean {}'.format(metric), xlabel='')
plt.legend(fontsize=12)
ax.set(xlim=(25, 55), xlabel='Time after induction (min)')
sns.despine()
plt.tight_layout()
plt.savefig(plot_dir+'Rep_meanmass.pdf', bbox_inches='tight')

# make replicate number column to show all data points in heatmap after pivot
#freq_df['rep'] = freq_df.groupby(['strain','thresh']).transform(lambda x: np.arange(len(x))).no_cells
#freq_df['rep'] = freq_df.groupby(['strain','thresh']).exp.transform(lambda x: x-x.min())
# Does fraction of active cells change over the 20 min?
# some correlation with time, less active cells...but conclusion is the same
fig, ax = plt.subplots(figsize=(10,6))
for strain, _freq_df in freq_df[freq_df.thresh==8].groupby('strain'):
    sns.regplot(x='time_postinduction', y='frac_active', data=_freq_df, ax=ax, color=colors_strain[strain], label=strain)
ax.set(ylabel='Active cells fraction', xlabel='Time after induction (min)')
plt.legend(handles=patches)
sns.despine()
plt.tight_layout()
plt.savefig(plot_dir+'Rep_fracActive.pdf', bbox_inches='tight')

# check distributions for modeling with qq-plot; need to do regression first
# just by eyeballing, normal distrib does not capture tails
# nbinom seems like best description, but params are not very intuitive...
import scipy.stats as stats
fig, axes = plt.subplots(len(strains2plot))
fit = []
for ax, (strain, group) in zip(axes.ravel(), parts[(parts.mass_norm>=2)&(parts.corrwideal>=0.5)&(parts.strain.isin(strains2plot))].groupby('strain')):
    values = group.mass_norm.values
    fit.append((strain, stats.probplot(values, dist="geom", sparams=(0.3), plot=ax)[1]))

# Summary plot
df2plot =  parts[(parts.mass_norm>=8)&(parts.corrwideal>=0.5)&(parts.strain.isin(strains2plot))]
order = sorted(df2plot.CTDr.unique())
with sns.axes_style(*style):
    fig, ax = plt.subplots(figsize=(9, 6))
# pooled
sns.pointplot(x='strain', y='mass_norm', order=strains2plot,
        data=df2plot, join=False,
        estimator=np.median, ci=99, ax=ax)
# each image
sns.stripplot(x='strain', y='mass_norm', order=strains2plot,
        data=parts[(parts.mass_norm>=7)&(parts.corrwideal>=0.5)&(parts.strain.isin(strains2plot))].groupby(['mov_name', 'strain']).median().reset_index(),
        ax=ax, size=12, alpha=0.25)
ax.set(ylabel='Median TS intensity (a.u.)', xlabel='CTD repeats', ylim=(ax.get_yticks()[0]-0.2, ax.get_yticks()[-1]))
sns.despine(left=False, bottom=False)
plt.tight_layout()

# check peaks found
from skimage import io
import trackpy as tp
imname = '03052019_yQC21_255u100%int480_150msExp_30-45minPosGal_6_w2GFPlow'
imname = '03272019_TL47pQC99_255u100%int480_150msExp_30-50minPosGal_13_w2GFPlow'
imname = '04162019_TL47pQC119S2_255u100%int480_150msExp_30-50minPosGal_1_w2GFPlow'
im_dir='/Users/porfirio/lab/yeastEP/ms2pp7/data/2019_pp7Snapshots/{}'.format('_'.join(imname.split('_')[:2]))
im = io.imread('{}/{}.tif'.format(im_dir, imname))
fig, ax = plt.subplots()
tp.annotate(parts[(parts.mass_norm>=3)&(parts.mov_name==imname)], im, color='w')
tp.annotate(parts[(parts.mass_norm>=7)&(parts.corrwideal>=0.5)&(parts.mov_name==imname)], np.log(im))

fig, axes = plt.subplots(2,4, sharex=True, sharey=True)
for ax,t in zip(axes.ravel(), np.arange(1,9)):
    parts_percell = parts[(parts.mass_norm>=t)&(parts.corrwideal>=0.5)].groupby(['cid', 'mov_name','strain']).count().x.reset_index()
    parts_percell.groupby(['strain']).apply(lambda x: ax.scatter(*ecdf_step(x.x.values), color=colors_strain[x.name], alpha=0.3, s=60, label=x.name))
    #parts_percell.groupby(['strain','mov_name']).apply(lambda x: ax.scatter(*ecdf_step(x.x.values), color=colors_strain[x.name[0]], alpha=0.3, s=60, label=x.name[0]))
    ax.set_title(t)
plt.legend(handles=patches, fontsize=10)
sns.despine()

# massnorm 7 seems to be a good threshold; get df with counts by image and strain
# count spots per cell
parts_percell = parts[(parts.mass_norm>=7)&(parts.corrwideal>=0.5)].groupby(['cid', 'mov_name','strain']).count().x.reset_index()
parts_percell = parts_percell.rename(columns={'x':'num_spots'})
# number of spots vs cumulative fraction of cells per image
parts_percell_frac = parts_percell.groupby(['mov_name','strain']).apply(lambda x: pd.Series(ecdf_step_r(x.num_spots.values))).reset_index()
# tidy up
tidy_ppcell, max_spots = [], parts_percell.num_spots.max()
for _, row in parts_percell_frac.iterrows():
    for i in np.arange(max_spots):
        try:
            tidy_ppcell.append((row.mov_name, row.strain, row[0][i], row[1][i]))
        except IndexError: # for clarity, frac=0.0 if no cells have this number
            tidy_ppcell.append((row.mov_name, row.strain, i+1, 0.0))

tidy_ppcell = pd.DataFrame(tidy_ppcell)
tidy_ppcell.columns = ['mov_name','strain','num_spots','frac']
# now can plot
strains_recr = ['yQC21', 'TL47pQC116', #control and 13rCTD 
        'TL47pQC99', 'TL47pQC1192E', 'TL47pQC1195D', 'TL47pQC119S2', #FUS and mutants
        'TL47pQC115', 'TL47pQC1202H','TL47pQC1203K'] #TAF and mutants
        #'TL47pQC118'] # HP1a
colors_recr = cm.magma(np.linspace(0.2,1, len(strains_recr)))
colors_recr = {s:c for s,c in zip(strains_recr, colors_recr)}
patches = [mpatches.Patch(color=colors_recr[l], label=l) for l in strains_recr]

# boxplot
fig = sns.catplot(x='num_spots', y='frac', hue='strain', kind='box', legend=False,
        palette=colors_recr, hue_order=strains_recr, data=tidy_ppcell,
        height=10, aspect=15/10)
# or stripplot, box looks better
#fig = sns.stripplot(x='num_spots', y='frac', hue='strain', dodge=True, size=8,
#        palette=colors_recr, hue_order=strains_recr, data=tidy_ppcell)
# draw horizontal line for comparison to 13rCTD
medianprops = dict(linestyle='--', linewidth=2, color='#385a7c', alpha=0.3)
# easier to use boxplot func for horizontal line, hide everything but median
sns.boxplot(x='num_spots', y='frac', whis=0, fliersize=0, linewidth=0,
        data=tidy_ppcell[tidy_ppcell.strain=='TL47pQC116'],
        medianprops=medianprops, boxprops=dict(alpha=0), ax=fig.ax)
fig.ax.set(xlabel='nucleation sites', ylabel='fraction of cells',
        xlim=(-0.5, 5.5), xticks=np.arange(0,6))
fig.ax.set_xticklabels(np.arange(1,7))
fig.ax.legend(handles=patches, fontsize=10)
plt.savefig('../output/pipeline_snapshots/plots/recruitment/nucleationsites.pdf', bbox_inches='tight')



fig, ax = plt.subplots()
parts_percell.groupby(['strain']).apply(lambda x: ax.scatter(*ecdf_step(x.num_spots.values), color=colors_strain[x.name], alpha=0.8, s=100, label=x.name, marker='_'))
parts_percell.groupby(['mov_name']).apply(lambda x: ax.scatter(*ecdf_step(x.num_spots.values), color=colors_strain[x.strain.iloc[0]], alpha=0.8, s=100, label=x.strain.iloc[0], marker='_'))

def ecdf_step(data):
    """ get steps only of an ecdf for `data`"""
    x = np.unique(data)
    y = [np.sum(data<=_x)/len(data) for _x in x]
    return x,y

def ecdf_step_r(data):
    """ get fraction of dataset at each value """
    x = np.unique(data)
    y = [np.sum(data==_x)/len(data) for _x in x]
    return x,y
