from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import pickle
from im_utils import *

# load dataframes
#ddir = '../output/20171122/'
#spot_df = pd.DataFrame()
#for strain in os.listdir(ddir):
#    spot_df = pd.concat((spot_df, pd.read_csv(ddir+strain)))
## put strain name in new column
#spot_df['strain'] = spot_df.imname.apply(lambda x: x.split('FISH')[0])

# discard bad samples (seems like didn't wash so well, in edge of coverslip)
#spot_df = spot_df[~spot_df.imname.isin(('67f1FISHGal10PP7_s13', '664FISHGal10PP7_s10'))]

#TODO:

#spot_df['mrnas'] = mrnas
#spot_df['mrnas'] = spot_df.mass / mrna_df.mass.median()
#ints = spot_df.apply(lambda x: gauss3d(X, (0, x.a, x.bx, x.by, x.bz, x.cx, x.cy, x.cz)).sum(), axis=1)

spot_df = pd.read_csv('../output/mrnaquant_20171201_succesfulRegressFloatFullZ.csv')
# remove false spots and spots outside cells
spot_df_mrnas = spot_df[(spot_df.svm_label.isin(['mrna', 'TS']))&(spot_df.cell_label>0)]

# count transcripts by cell
transcripts_bycell = spot_df_mrnas.groupby('cid').no_mrnas.sum().reset_index()
transcripts_bycell['strain']  = transcripts_bycell.cid.apply(lambda x: x.split('FISH')[0])

# get empty cells
# TODO: empty cells should now be in spot_df dataframe
seg_coords['strain']  = seg_coords.imname.apply(lambda x: x.split('FISH')[0])
cellnums = seg_coords.groupby('strain').is_nucleus.sum().reset_index()
cells_wrna = transcripts_bycell.groupby('strain').no_mrnas.count().reset_index()
total_cells = pd.merge(cellnums, cells_wrna, on='strain')
total_cells['empty']  = (total_cells.is_nucleus - total_cells.no_mrnas).astype(int)

#fig, axes = plt.subplots(2,2)
#axes = iter(np.ravel(axes))
fig, ax = plt.subplots(1)
for name, group in transcripts_bycell.groupby('strain'):
    #ax=next(axes)
    #mrnas by cell
    mrnas_bycell = group.no_mrnas.values
    # empty cells
    empty_cells = np.full(total_cells[total_cells.strain==name]['empty'].values, 0)
    # complete distribution
    mrnas_bycell = np.round(np.concatenate((mrnas_bycell, empty_cells)))
    #sns.distplot(group.mrnas, ax=ax, bins=50)
    label='{0} mean {1:0.2f} median {2:0.2f}'.format(name, np.mean(mrnas_bycell), np.median(mrnas_bycell))
    #plot_ecdf(mrnas_bycell, ax, label=label, formal=0)
    plot_ecdf(mrnas_bycell, ax, label=label, formal=0, alpha=0.8)
    ax.set_title(label)
plt.tight_layout()
plt.legend()

#sns.stripplot(x='no_mrnas', y='strain', data=spot_df_mrnas, alpha=0.1)
#plt.tight_layout()


from scipy.special import gamma as gamma_f
from scipy.special import hyp1f1
import emcee

def raj_model(m, lrat, grat, mdel):
    t1 = gamma_f(lrat+m) / (gamma_f(m+1) * gamma_f(lrat+grat+m))
    t2 = gamma_f(lrat+grat) / gamma_f(lrat)
    t3 = (mdel)**m
    t4 = hyp1f1(lrat+m, lrat+grat+m, -mdel)
    return np.log(t1*t2*t3*t4)

n_dim = 2        # number of parameters in the model (r and p)
n_walkers = 50   # number of MCMC walkers
n_burn = 1000    # "burn-in" period to let chains stabilize
n_steps = 2000   # number of MCMC steps to take after burn-in
sampler = emcee.EnsembleSampler(n_walkers, n_dim, raj_model, 
                                args=(simul, r_max), threads=6)

mu = 1.4 # mrna per minute
delta = 1/20 # mrna per minute
mdel = mu/delta
#lrat = lam/delta
#grat = gamma/delta

with pm.Model() as model:
    # Priors
    lam = pm.Uniform('lam', lower=0, upper=100)
    gamma = pm.Uniform('gamma', lower=0, upper=100)
    # Likelihood
    obs = pm.Poisson('obs', mu=lfreq, observed=simul)
    trace = pm.sample(draws=1000, tune=1000, njobs=4)


################################################################################

# count transcripts by cell
spot_df['cellid'] = spot_df['imname'] + spot_df.cell_label.apply(str)
transcripts_bycell = spot_df[spot_df.svm_label=='mrna'].groupby('cellid').x.count().reset_index()
transcripts_bycell['strain']  = transcripts_bycell.cellid.apply(lambda x: x.split('FISH')[0])
c = iter(sns.color_palette("Set2", 10))
for name, group in transcripts_bycell.groupby('strain'):
    _median, _mean = np.median(group.x), np.mean(group.x)
    label = '{0} median: {1:.2f} mean: {2:.2f}'.format(name, _median, _mean)
    plt.plot(*ecdf(group.x, conventional=True), c=next(c), label=label)
    #ax.set_title(label)
plt.legend()

# TS int
for name, group in spot_df[(spot_df.svm_label=='TS')&(spot_df.mass>00)].groupby('imname'):
    #for name, group in TS_certain.groupby('strain'):
    _median, _mean = np.median(group.mass), np.mean(group.mass)
    plot_ecdf(group.mass, label='{0} median: {1:.2f} mean: {2:.2f}'.format(name, _median, _mean), formal=1)
plt.legend()

# Distribution of mrna int by sample
for name, group in spot_df[spot_df.svm_label=='mrna'].groupby('imname'):
    #if '68' in name: continue
    _median, _mean = np.median(group.mass), np.mean(group.mass)
    label = '{0} median: {1:.2f} mean: {2:.2f}'.format(name, _median, _mean)
    print(label)
    plot_ecdf(group.mass, label=label, formal=0)
plt.legend()



fig, axes = plt.subplots(1)
plot_ecdf(spot_df[spot_df.svm_label=='TS'].cx, ax=axes)
plot_ecdf(spot_df[spot_df.svm_label=='mrna'].cx, ax=axes)
plot_ecdf(spot_df[spot_df.svm_label=='crap'].cx, ax=axes)
plt.scatter(spot_df[spot_df.svm_label=='TS'].nuc_label.values>0,spot_df[spot_df.svm_label=='TS'].mass, alpha=0.1)

s=100
TS_certain = spot_df.sort_values('signal', ascending=False).groupby('cellid').apply(lambda x: x.iloc[0])
TS_certain = TS_certain[TS_certain.apply(lambda x: check_borders(x[['x','y']],
                                            ims_smooth[x.imname], s), axis=1)]
ts_certain_im = TS_certain.apply(lambda x: [get_bbox(x[['x','y']],
                            ims_smooth[x.imname], s)], axis=1)
ts_certain_im   = np.stack([i[0] for i in ts_certain_im ])
plt.figure()
io.imshow(im_block(ts_certain_im, 50, norm=0), cmap='viridis')

# look at SVM labels
_ims = get_batch_bbox(spot_df[spot_df.svm_label=='mrna'], ims_proj)
plt.figure()
io.imshow(im_block(_ims, 30, norm=1), cmap='viridis')
