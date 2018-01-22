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
ims_proj = load_ims(rrdir+'zprojected/', 'tif', channel=0)
# clear peaks near border
size=9
spot_df_ = spot_df[spot_df.apply(lambda x: check_borders(x[['x','y']],
                    ims_proj[x.imname], size), axis=1)].reset_index(drop=True)
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

# get all spot images
#ts_df = spot_df[spot_df.svm_label=='TS'].copy().reset_index(drop=True)
spot_df = pd.read_csv('../output/{}_smFishPeaks3D.csv'.format('20171201'))
ts_ims = get_batch_bbox(spot_df, ims_stack, wsize, im3d=True, size_z='Full')
# convert to float for regression
ts_ims_float = [skimage.img_as_float(im) for im in ts_ims]

# Dataframe to store optimized params and error bars by field of view
popt_all = pd.DataFrame()
# Initial guesses for regression. Center is roughly center of image.
# Params are: r, a, bx, by, bz, cx, cy, cz
# r is offset, a is coeff, b are center coord, c are sigmas
p0 = (np.median(np.concatenate([i.ravel() for i in ts_ims_float])), 0.05, *((wsize/2),)*3, 2, 2, 2)
# Parameter bounds, the same for all b's and c's
p_range = ((0,1), (0, 1), *((0, wsize),)*6)
for i, ts_im in tqdm(enumerate(ts_ims_float)):
    try: #minimizing the negative log posterior
        _popt, _err =  imstack_gauss3d_regress(ts_im, p0, p_range)
    except: #least of squares is more robust
        try:
            _popt =  imstack_gauss3d_regress(ts_im, p0, leastsq=1)
        except:
            _popt = np.full_like(p0, np.nan)
    # Add to dataframe
    _popt_df = pd.DataFrame(_popt).T
    popt_all = pd.concat((popt_all, _popt_df), ignore_index=True)
popt_all.columns = ('r', 'a', 'bx', 'by', 'bz','cx','cy','cz')
popt_all_cols = ['r', 'a', 'bx', 'by', 'bz','cx','cy','cz']
popt_all = spot_df[popt_all_cols]

# check fits
#popt_all = spot_df[['r', 'a', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']].copy().reset_index(drop=True)
wsize=9
X = np.array([d.ravel() for d in np.indices((wsize, wsize, wsize))])
ts_ims_fit = np.stack([gauss3d(X, im[1]).reshape((-1, wsize, wsize)) for im in popt_all.iterrows()])
fit_ints = popt_all.apply(lambda x: gauss3d(X, x).sum() - x.r*wsize**3, axis=1)
spot_df['gauss_int'] = fit_ints
# best to sort are: cx/yz, a, bx, by
#sortind = spot_df.sort_values(['no_mrnas']).index
sortind = spot_df[(spot_df.svm_label.isin(['mrna','TS']))].sort_values(['mass']).index
ims_fit = np.stack([z_project(im) for im in ts_ims_fit[sortind]])
#ims_raw = np.stack([z_project(im) for im in ts_ims])[sortind]
ims_raw = np.stack([im for im in spot_ims])[sortind]

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
axes[0].imshow(im_block(ims_fit, 60, norm=0), cmap='viridis')
axes[1].imshow(im_block(ims_raw, 60, norm=0), cmap='viridis')
plt.tight_layout()

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
#axes[0].imshow(im_block(ims_fit, 150, norm=0), cmap='viridis')
axes[0].imshow(np.clip(im_block(ims_fit, 50, norm=0), 0, 0.005), cmap='viridis')
axes[1].imshow(np.clip(im_block(ims_raw, 50, norm=0), 0, 10000), cmap='viridis')

# manual Quality Assurance
spot_df['pid'] = spot_df.apply(lambda x: str(int(x.x))+str(int(x.y))+x.imname, axis=1)
mrna_qa = spot_df[(spot_df.svm_label.isin(['crap']))].sort_values(['corrwideal']).copy()
ind_mrna, ims_mrnaqa, mrna_qa_ = sel_training(mrna_qa, ims_proj, ncols=200, scale=1, mark_center=0, s=9, cmap='viridis')
# create new label for these in each round: crap_mrna, crap_mrna2...
spot_df.loc[spot_df.pid.isin(mrna_qa[ind_mrna].pid), 'svm_label'] = 'crap_TS'

# compute correlation to an ideal spot: point source with gaussian of PSF width
# get all 3d images
ims_stack = load_ims(rrdir+'zstacks/', 'STK')
# get all spot images
allims = get_batch_bbox(spot_df, ims_stack, 9, im3d=True, size_z='Full')

#allims = get_batch_bbox(spot_df, ims_proj)
idealspot = np.full((9,9,9), 0)
idealspot[4,4,4] = 1 # single light point source
idealspot = skimage.filters.gaussian(idealspot, sigma=4.2) # PFS width blur
# compute pearson corr; very similar result obtained with spearmanr
#allcorrs = [np.corrcoef(idealspot.ravel(), im.ravel())[1][0] for im in allims]
allcorrs = []
allcorrs_s = []
# do try/except bcause of different z-depths
for im in allims:
    idealspot = np.full_like(im, 0)
    center = (idealspot.shape[0]//2,idealspot.shape[1]//2,idealspot.shape[2]//2)
    idealspot[center] = 1 # single light point source
    idealspot = skimage.filters.gaussian(idealspot, sigma=4.2) # PFS width blur
    allcorrs_s.append(scipy.stats.spearmanr(idealspot.ravel(), im.ravel())[0])
    allcorrs.append(np.corrcoef(idealspot.ravel(), im.ravel())[1][0])
spot_df['corrwideal'] = allcorrs
spot_df['corrwideal_s3d'] = allcorrs_s

fig, ax = plt.subplots(1)
for l in spot_df.svm_label.unique():
    plot_ecdf(spot_df[spot_df.svm_label==l].corrwideal_s3d, label=l, ax=ax)
plt.legend()

# do it in steps
mrna_qa = spot_df[(spot_df.svm_label.isin(['crap']))].sort_values(['cx']).copy()
ind_r, ims_r, mrna_qa_ = [], [], pd.DataFrame()
step = 2000
for n in np.arange(step, len(mrna_qa)+step, step):
    _ind_mrna, _ims_mrnaqa, _mrna_qa = sel_training(mrna_qa[n-step:n],
        ims_proj, ncols=60, mark_center=0, s=9, cmap='viridis', normall=1)
    print(n-step, n)
    ind_r.append(_ind_mrna)
    ims_r.append(_ims_mrnaqa)
    mrna_qa_ = pd.concat((mrna_qa_, _mrna_qa), ignore_index=True)
ind_r = np.concatenate(ind_r)
# create new label for these in each round: crap_mrna, crap_mrna2...
spot_df.loc[spot_df.pid.isin(mrna_qa_[ind_r].pid), 'svm_label'] = 'former_crap'



# compute TS intensities; substract background by substracting offset ('r')
fit_ints = popt_all.apply(lambda x: gauss3d(X, x).sum() - x.r*wsize**3, axis=1)

# compute average mRNA intensity using good-bad data model
mrna_ind = spot_df.svm_label=='mrna'
mrna_int = fit_ints[mrna_ind]
with pm.Model() as model:
    # Priors
    mu = pm.Uniform('mu', 0, 10)
    sigma = bebi103.pm.Jeffreys('sigma', 0.1, 1)
    sigma_bad = bebi103.pm.Jeffreys('sigma_bad', sigma, 10)
    w = pm.Beta('w', alpha=0.5, beta=0.5, shape=len(mrna_int))
    # Likelihood is good-bad data model.
    a_obs = bebi103.pm.GoodBad('a_obs',
                               mu=mu,
                               sigma=sigma,
                               sigma_bad=sigma_bad,
                               w=w,
                               observed=mrna_int)
    trace_goodbad = pm.sample(draws=2000, tune=2000, njobs=4)

df_mcmc = bebi103.pm.trace_to_dataframe(trace_goodbad)
# get prob of being bad (high w -> bad, low w -> good)
wcols = [c for c in df_mcmc.columns if 'w_' in c]
w = np.median(df_mcmc[wcols].values, axis=0)
plt.figure()
corner.corner(df_mcmc[['mu', 'sigma', 'sigma_bad']])
# smaller valued intensities have higher prob of being bad
plt.scatter(w, mrna_int, s=10, alpha=0.2)
# add bad prob and filter
mrna_int['w_bad'] = w
mrna_int_good = mrna_int[(mrna_int['w_bad']<0.2)]


# compute mrnas per TS and add to DF
mrnas = fit_ints/mrna_fit_int

spot_df = pd.concat((spot_df, popt_all), axis=1)
#TODO:
#spot_df.to_csv('../output/mrnaquant_20171201_succesfulRegressFloatFullZ.csv', index=False)
spot_df = pd.read_csv('../output/mrnaquant_20171201_succesfulRegressFloatFullZ.csv')

#spot_df['mrnas'] = mrnas
#spot_df['mrnas'] = spot_df.mass / mrna_df.mass.median()
#ints = spot_df.apply(lambda x: gauss3d(X, (0, x.a, x.bx, x.by, x.bz, x.cx, x.cy, x.cz)).sum(), axis=1)
smrna_int = spot_df[spot_df.svm_label=='mrna'].gauss_int.median()
smrna_mass = spot_df[spot_df.svm_label=='mrna'].mass.median()
spot_df['no_mrnas'] = np.round(spot_df.gauss_int/smrna_int)
spot_df['no_mrnas'] = np.round(spot_df.mass/smrna_mass)
spot_df.loc[(spot_df.no_mrnas<1), 'no_mrnas'] = 1

with open('../output/nuc_trainingset/20171201_manualsel_nuclei_segment_centroids_markers_segim.pkl', 'rb') as f:
    seg_coords = pickle.load(f)
    sel_markers_dict = pickle.load(f)

# update cell labels
spot_df['cell_label'] = spot_df.apply(lambda coords:\
    sel_markers_dict[coords.imname][0][int(coords.y), int(coords.x)], axis=1)
# Get nuclear label
spot_df['nuc_label'] = spot_df.apply(lambda coords:\
    sel_markers_dict[coords.imname][1][int(coords.y), int(coords.x)], axis=1)

spot_df['strain'] = spot_df.imname.apply(lambda x: x.split('FISH')[0])
spot_df['cid'] = spot_df.apply(lambda x: x.imname+'_'+str(x.cell_label), axis=1)
spot_df['nid'] = spot_df.apply(lambda x: x.imname+'_'+str(x.nuc_label), axis=1)
# remove false spots and spots outside cells
spot_df_mrnas = spot_df[(spot_df.svm_label.isin(['mrna', 'TS']))&(spot_df.cell_label>0)]

# count transcripts by cell
transcripts_bycell = spot_df_mrnas.groupby('cid').no_mrnas.sum().reset_index()
transcripts_bycell['strain']  = transcripts_bycell.cid.apply(lambda x: x.split('FISH')[0])

# get empty cells
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
