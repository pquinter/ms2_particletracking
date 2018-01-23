from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#========================================================================
# Fit spots to 3d Gaussian
#========================================================================
# get all 3d images
ims_stack = load_ims(rrdir+'zstacks/', 'STK')

# get all spot images
spot_df = pd.read_csv('../output/{}_smFishPeaks3D.csv'.format('20171201'))
spot_ims = get_batch_bbox(spot_df, ims_stack, wsize, im3d=True, size_z='Full')
# convert to float for regression
spot_ims_float = [skimage.img_as_float(im) for im in spot_ims]

# Dataframe to store optimized params and error bars by field of view
popt_all = pd.DataFrame()
# Initial guesses for regression. Center is roughly center of image.
# Params are: r, a, bx, by, bz, cx, cy, cz
# r is offset, a is coeff, b are center coord, c are sigmas
p0 = (np.median(np.concatenate([i.ravel() for i in spot_ims_float])), 0.05, *((wsize/2),)*3, 2, 2, 2)
# Parameter bounds, the same for all b's and c's
p_range = ((0,1), (0, 1), *((0, wsize),)*6)
for i, ts_im in tqdm(enumerate(spot_ims_float)):
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
#popt_all_cols = ['r', 'a', 'bx', 'by', 'bz','cx','cy','cz']
#popt_all = spot_df[popt_all_cols]
spot_df = pd.concat((spot_df, popt_all), axis=1)
# compute intensities and subtract background (r)
fit_ints = popt_all.apply(lambda x: gauss3d(X, x).sum() - x.r*wsize**3, axis=1)
spot_df['gauss_int'] = fit_ints
#spot_df.to_csv('../output/mrnaquant_20171201_succesfulRegressFloatFullZ.csv', index=False)
spot_df = pd.read_csv('../output/mrnaquant_20171201_succesfulRegressFloatFullZ.csv')

# check fits
wsize=9
X = np.array([d.ravel() for d in np.indices((wsize, wsize, wsize))])
spot_ims_fit = np.stack([gauss3d(X, im[1]).reshape((-1, wsize, wsize)) for im in popt_all.iterrows()])
# best to sort are: cx/yz, a, bx, by
#sortind = spot_df.sort_values(['no_mrnas']).index
sortind = spot_df[(spot_df.svm_label.isin(['mrna','TS']))].sort_values(['mass']).index
ims_fit = np.stack([z_project(im) for im in spot_ims_fit[sortind]])
#ims_raw = np.stack([z_project(im) for im in spot_ims])[sortind]
ims_raw = np.stack([im for im in spot_ims])[sortind]

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
axes[0].imshow(im_block(ims_fit, 60, norm=0), cmap='viridis')
axes[1].imshow(im_block(ims_raw, 60, norm=0), cmap='viridis')
plt.tight_layout()

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
axes[0].imshow(np.clip(im_block(ims_fit, 50, norm=0), 0, 0.005), cmap='viridis')
axes[1].imshow(np.clip(im_block(ims_raw, 50, norm=0), 0, 10000), cmap='viridis')

# compute correlation to an ideal spot: point source blurred with gaussian of PSF width
allims = get_batch_bbox(spot_df, ims_proj)
idealspot = np.full((9,9), 0)
idealspot[4,4] = 1 # single light point source
idealspot = skimage.filters.gaussian(idealspot, sigma=4.2) # PSF width blur
# pearson corr; tried spearman and 3d Corr, pearson on 3d proj is best
allcorrs = [np.corrcoef(idealspot.ravel(), im.ravel())[1][0] for im in allims]
spot_df['corrwideal'] = allcorrs


# compute average mRNA intensity using good-bad data model
# TODO: try correlation to ideal spot instead of intensities?
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
