from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from im_utils import plot_ecdf
from matplotlib import cm
import matplotlib.patches as mpatches

part_dir = '../output/pipeline_snapshots/particles'
parts = pd.read_csv(part_dir+'/parts_labeled.csv')
parts = parts[~(parts.strain=='yQC25')]
rep_dict = {'yQC21':26, 'yQC22':14,'yQC23':12,'yQC62':10, 'yQC26':10.1,
        'yQC63TL174':9, 'TL47pQC99':33, 'TL47pQC116':30 , 'TL47pQC115':32, 'TL47pQC118':31}
parts['CTDr'] = parts.strain.map(rep_dict)

################################################################################
################################################################################
plot_dir = '../output/pipeline_snapshots/plots/'
colors = ['b','k','r','seagreen','mediumorchid', 'g','y','cyan']
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
    try: colors_r = cmap(np.linspace(0.1,1, len(CTDr)))
    except TypeError: colors_r = cmap
    if isinstance(cmap, str): cmap = [cmap]*len(CTDr)
    colors_ctd = {r:c for r,c in zip(CTDr, colors_r)}
    patches_ctd = [mpatches.Patch(color=colors_ctd[l], label=int(l)) for l in colors_ctd]

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
colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0']
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
fig, axes = multi_ecdf(parts[(parts.mass_norm>=8)&(parts.corrwideal>=0.5)], strains2plot, plot_median=False)
            save=plot_dir+'burstsize/ecdf_size_sCTD.pdf')

# single plot
fig, ax = plt.subplots()
grouped = parts[(parts.mass_norm>=7)&(parts.corrwideal>=0.5)].groupby(['CTDr', 'mov_name'])
grouped.apply(lambda x: plot_ecdf(x.mass_norm.values, alpha=0.5,
                                color=colors_ctd[x.name[0]], ax=ax,
                                label=x.strain.iloc[0], formal=1, ylabel=''))
ax.set(ylabel='ECDF', xlabel='Burst size')
plt.tight_layout()
plt.legend(handles=patches_ctd, loc=4)
plt.savefig(plot_dir+'ECDF_byStrain.pdf', bbox_inches='tight')



# plot nuclear fluorescence over time

ind_times = parts.time_postinduction.unique()
colors_time = cm.magma(np.linspace(0,0.8, len(ind_times)))
colors_time = {t:c for t,c in zip(sorted(ind_times), colors_time)}

fig, ax = plt.subplots()
parts.drop_duplicates(['mov_name','roi']).groupby(['strain', 'time_postinduction']).apply(lambda x: ax.scatter(x.name[1], x.nuc_fluor.mean(), color=colors_strain[x.name[0]], label=x.name[0]))
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
tp.annotate(parts[(parts.mass_norm<=8)&(parts.mov_name==im_name)], np.log(im), ax=ax)
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
    for strain, sdf in parts[(parts.mass_norm>7)&(parts.corrwideal>=0.5)].groupby('strain'):
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
df2plot =  parts[(parts.mass_norm>=7)&(parts.corrwideal>=0.5)&(parts.strain.isin(strains2plot))]
order = sorted(df2plot.CTDr.unique())
with sns.axes_style(*style):
    fig, ax = plt.subplots(figsize=(9, 6))
# pooled
sns.pointplot(x='CTDr', y='mass_norm', order=order,
        data=df2plot,
        estimator=np.median, ci=99, ax=ax, palette=colors_ctd)
# each image
sns.stripplot(x='CTDr', y='mass_norm', order=order,
        data=parts[(parts.mass_norm>=7)&(parts.corrwideal>=0.5)&(parts.strain.isin(strains2plot))].groupby('mov_name').median().reset_index(),
        ax=ax, palette=colors_ctd, size=12, alpha=0.25)
ax.set(ylabel='Median TS intensity (a.u.)', xlabel='CTD repeats', ylim=(ax.get_yticks()[0]-0.2, ax.get_yticks()[-1]))
sns.despine(left=False, bottom=False)
plt.tight_layout()

# check peaks found
from skimage import io
import trackpy as tp
imname = '03052019_yQC21_255u100%int480_150msExp_30-45minPosGal_6_w2GFPlow'
imname = '03272019_TL47pQC99_255u100%int480_150msExp_30-50minPosGal_13_w2GFPlow'
im = io.imread('/Users/porfirio/lab/yeastEP/ms2pp7/data/2019_pp7Snapshots/03272019_TL47pQC99/{}.tif'.format(imname))
tp.annotate(parts[(parts.mov_name==imname)], im, color='w')
tp.annotate(parts[(parts.mass_norm>=8)&(parts.corrwideal>=0.5)&(parts.mov_name==imname)], np.log(im))
