from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Check Gaussian 3D fits
spot_df = pd.read_csv('../output/mrnaquant_20171201_succesfulRegressFloatFullZ.csv')
wsize=9
X = np.array([d.ravel() for d in np.indices((wsize, wsize, wsize))])
spot_ims_fit = np.stack([gauss3d(X, im[1]).reshape((-1, wsize, wsize)) for im in popt_all.iterrows()])

ims_proj_fish = load_ims(rrdir+'zprojected/', 'tif', channel=0)
spotims = get_batch_bbox(spot_df, ims_proj_fish)

# best to sort are: corrwideal, cx/yz, a, bx, by
sortind = spot_df[(spot_df.svm_label.isin(['mrna','TS']))].sort_values(['corrwideal']).index
sortind = spot_df[(spot_df.corrwideal>0.5)].sort_values(['corrwideal']).index
#ims_fit = np.stack([z_project(im) for im in spot_ims_fit[sortind]])
#ims_raw = np.stack([z_project(im) for im in spot_ims])[sortind]
ims_raw = np.stack([im for im in allims])[sortind]

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
    plot_ecdf(spot_df[spot_df.svm_label==l].corrwideal, label=l, ax=ax, formal=1, alpha=1)
plt.legend()
