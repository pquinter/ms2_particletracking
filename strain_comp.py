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

# load classifier
with open('../output/spot_trainingset/clf_beta1_size9NormedNotSmooth.pkl', 'rb') as f:
    clf = pickle.load(f)

# load spots df
spot_df = pd.read_csv('../output/{}_smFishPeaks3D.csv'.format('20171201'))
spot_df['strain'] = spot_df.imname.apply(lambda x: x.split('FISH')[0])
# Get images
rrdir = '../data/FISH/20171201/'
#ims_proj = load_ims(rrdir+'zprojected/', 'tif', channel=0)
# clear peaks near border
size=9
spot_df = spot_df[spot_df.apply(lambda x: check_borders(x[['x','y']],
                                        ims_proj[x.imname], size), axis=1)]
# get spot images
spot_ims = get_batch_bbox(spot_df, ims_proj, size=size)
# normalize and ravel for classification
spot_ims  = normalize_im(spot_ims)
spot_ims = np.stack([np.ravel(i) for i in spot_ims])
# classify
labels_pred = clf.predict(spot_ims)
# add labels
spot_df['svm_label'] = labels_pred
# discard bad samples (seems like didn't wash so well, in edge of coverslip)
#spot_df = spot_df[~spot_df.imname.isin(('67f1FISHGal10PP7_s13', '664FISHGal10PP7_s10'))]

#========================================================================
# Quantify transcripts per TS
#========================================================================
# get all 3d images
ims_stack = load_ims(rrdir+'zstacks/', 'STK')

# size of image window to get around predicted center
wsize =  11
bbox_imsstack = spot_df.apply(lambda x: [get_bbox3d(x[['x','y','z']],
                    size=wsize, im=ims_stack[x.imname])], axis=1)
# clear those not in border
notinborder = bbox_imsstack.apply(lambda x: np.array_equal(x[0].shape,
                (wsize, wsize, wsize)))
spot_df = spot_df[notinborder].copy().reset_index(drop=True)

# get mrna ims
mrna_df = spot_df[spot_df.svm_label=='mrna']
mrna_ims = get_batch_bbox(mrna_df, ims_stack, wsize, im3d=True)
# convert to float, otherwise regression fails
mrna_ims_float = skimage.img_as_float(mrna_ims)
# average mrna image
mrna_av = np.mean(mrna_ims_float, axis=0)

# Initial guesses for regression. Center is roughly center of image.
# Params are: r, a, bx, by, bz, cx, cy, cz
p0 = (np.median(mrna_av), np.median(mrna_av), *((wsize/2),)*3, 2, 2, 2)
# Parameter bounds, the same for all b's and c's
p_range = ((0,1), (0, 1), *((0, wsize),)*6)


## Params are: r, a, bx, by, bz, cx, cy, cz
## Initial guesses for regression. Center is roughly center of image.
#p0 = (100, 100, *((wsize/2,)*3), 2, 2, 2)
## Least of squares regression on residuals
popt3d, errbars = imstack_gauss3d_regress(mrna_av, p0, p_range)
# substract background by setting offset to 0
popt3d[0] = 0

X = np.array([d.ravel() for d in np.indices(mrna_av.shape)])
checkGaussFit3d(mrna_av, X, popt3d)
mrna_fit_int = gauss3d(X, popt3d).sum()

# get all images
ts_df = spot_df[spot_df.svm_label=='TS'].copy().reset_index(drop=True)
ts_ims = get_batch_bbox(ts_df, ims_stack, wsize, im3d=True)
# convert to float, otherwise regression fails
ts_ims_float = skimage.img_as_float(ts_ims)


# Dataframe to store optimized params and error bars by field of view
popt_all = pd.DataFrame()
# Initial guesses for regression. Center is roughly center of image.
# Params are: r, a, bx, by, bz, cx, cy, cz
p0 = (np.median(ts_ims_float), 0.05, *((wsize/2),)*3, 2, 2, 2)
# Parameter bounds, the same for all b's and c's
p_range = ((0,1), (0, 1), *((0, wsize),)*6)
for i, ts_im in tqdm(enumerate(ts_ims_float)):
    # skip beads with misc errors in optimization (e.g. can't compute hessian)
    try:
        _popt, _err =  imstack_gauss3d_regress(ts_im, p0, p_range)#, leastsq=0, return_err=False)
    except:
        _popt = np.full_like(p0, np.nan)
    # Add to dataframe
    _popt_df = pd.DataFrame(_popt).T
    popt_all = pd.concat((popt_all, _popt_df), ignore_index=True)
popt_all.columns = ('r', 'a', 'bx', 'by', 'bz','cx','cy','cz')
# substract background by setting offset ('r') to zero --> not great
popt_all['r'] = 0

# see what makes images fail??? -> no striking visible difference
#failed = ts_ims[popt_all.cz<0].copy()
#failed = np.stack([z_project(im.T) for im in failed])
#plt.figure()
#plt.imshow(im_block(failed, 10, norm=1))
#
#succ = ts_ims[popt_all.cz>0].copy()
#succ = np.stack([z_project(im.T) for im in succ])
#plt.figure()
#plt.imshow(im_block(succ, 40, norm=1))

# compute TS intensities
fit_ints = popt_all.apply(lambda x: gauss3d(X, x).sum(), axis=1)
# compute mrnas per TS and add to DF
mrnas = fit_ints/mrna_fit_int

spot_df = pd.concat((spot_df, popt_all), axis=1)
#TODO:
# filter based on 3D gaussian fit!!
# train SVM including opt params?? probably would work much better
# use single class SVM
# - get number of cells, even those without transcripts!
#spot_dfbk.to_csv('../output/mrnaquant_20171201_test.csv', index=False)
spot_df = pd.read_csv('../output/mrnaquant_20171201_test.csv')

#spot_df['mrnas'] = mrnas
#spot_df['mrnas'] = spot_df.mass / mrna_df.mass.median()
#ints = spot_df.apply(lambda x: gauss3d(X, (0, x.a, x.bx, x.by, x.bz, x.cx, x.cy, x.cz)).sum(), axis=1)
spot_df['mrnas'] = ints/mrna_fit_int
spot_df['strain'] = spot_df.imname.apply(lambda x: x.split('FISH')[0])
sns.stripplot(x='mrnas', y='strain', data=spot_df[(spot_df.r>0)&(spot_df.mrnas>0)], alpha=0.01)
plt.tight_layout()

# count transcripts by cell
spot_df['cellid'] = spot_df['imname'] + spot_df.cell_label.apply(str)
transcripts_bycell = spot_df[(spot_df.r>0)&(spot_df.mrnas>0)&(spot_df.svm_label.isin(['mrna','TS']))].groupby('cellid').mrnas.sum(skipna=True).reset_index()
transcripts_bycell['strain']  = transcripts_bycell.cellid.apply(lambda x: x.split('FISH')[0])

transcripts_bycell = spot_df[spot_df.svm_label=='TS']
fig, axes = plt.subplots(2,2)
axes = iter(np.ravel(axes))
for name, group in transcripts_bycell.groupby('strain'):
    #for name, group in TS_certain.groupby('strain'):
    _median, _mean = np.median(group.mrnas), np.mean(group.mrnas)
    ax=next(axes)
    label='{0} mean {1:0.2f} median {2:0.2f}'.format(name, _mean, _median)
    #sns.distplot(group.mrnas, ax=ax, bins=50)
    plot_ecdf(np.round(group.mrnas), ax, label=label, formal=0)
    ax.set_title(label)
plt.tight_layout()
#plt.legend()



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



plot_ecdf(spot_df[spot_df.svm_label=='TS'].mass)
plot_ecdf(spot_df[spot_df.svm_label=='mrna'].mass)
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
