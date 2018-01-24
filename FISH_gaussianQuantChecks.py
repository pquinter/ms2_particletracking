from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Check Gaussian 3D fits
spot_df = pd.read_csv('../output/mrnaquant_20171201_succesfulRegressFloatFullZ.csv')
wsize=9
X = np.array([d.ravel() for d in np.indices((wsize, wsize, wsize))])
spot_ims_fit = np.stack([gauss3d(X, im[1]).reshape((-1, wsize, wsize)) for im in popt_all.iterrows()])
# best to sort are: corrwideal, cx/yz, a, bx, by
sortind = spot_df[(spot_df.svm_label.isin(['mrna','TS']))].sort_values(['corrwideal']).index
#ims_fit = np.stack([z_project(im) for im in spot_ims_fit[sortind]])
#ims_raw = np.stack([z_project(im) for im in spot_ims])[sortind]
ims_raw = np.stack([im for im in spot_ims])[sortind]

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
axes[0].imshow(im_block(ims_fit, 100, norm=0), cmap='viridis')
axes[1].imshow(im_block(ims_raw, 100, norm=1), cmap='viridis')
plt.tight_layout()

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
axes[0].imshow(np.clip(im_block(ims_fit, 50, norm=0), 0, 0.005), cmap='viridis')
axes[1].imshow(np.clip(im_block(ims_raw, 50, norm=0), 0, 500), cmap='viridis')

# correlation with ideal spot separates sets best, followed by cx, a
fig, ax = plt.subplots(1)
for l in spot_df.svm_label.unique():
    #    plot_ecdf(allcrosscorrs[spot_df.svm_label==l], label=l+'cross', ax=ax, formal=1, alpha=1)
    #    plot_ecdf(allcorrs[spot_df.svm_label==l], label=l, ax=ax, formal=1, alpha=1)
    plot_ecdf(spot_df[spot_df.svm_label==l].cx, label=l, ax=ax, formal=1, alpha=1)
plt.legend()

# compute average mRNA intensity using good-bad data model
# TODO: try correlation to ideal spot instead of intensities?: not good
mrna_ind = spot_df.svm_label=='mrna'

with pm.Model() as model:
    # Priors
    mu = pm.Uniform('mu', -1, 1)
    sigma = bebi103.pm.Jeffreys('sigma', 0.01, 0.5)
    sigma_bad = bebi103.pm.Jeffreys('sigma_bad', sigma, 1)
    w = pm.Beta('w', alpha=0.5, beta=0.5, shape=len(allcrosscorrs))
    # Likelihood is good-bad data model.
    a_obs = bebi103.pm.GoodBad('a_obs',
                               mu=mu,
                               sigma=sigma,
                               sigma_bad=sigma_bad,
                               w=w,
                               observed=allcrosscorrs)
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
