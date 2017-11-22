from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import sklearn

"""
Attempt to Classify spots using a machine learning
"""

#peaks = peaks.read_csv('../output/111617_TL47PP7smFish.csv')
# get 10by10 box containing candidate spots
def get_bbox(center, im, size=10, return_im=True):
    """ get 10 by 10 bounding box from center"""
    x, y = center
    s = int(size/2)
    x, y = int(x), int(y)
    bbox = np.s_[y-s:y+s, x-s:x+s]
    if return_im:
        return im[bbox]
    else: return bbox


peaks['uid'] = np.arange(len(peaks))
# get only peaks not in TS using unique id
TS_certain = peaks.sort_values('signal', ascending=False).groupby('cellid').apply(lambda x: x.iloc[0])
peaks_woTS = peaks[~(peaks.uid.isin(TS_certain.uid.values))]

peaks_woTS = peaks_woTS.apply(lambda x: [normalize_im(get_bbox(x[['x','y']],
                            ims[x.imname][:,:,0]))], axis=1)
# take a look at spots
peaks_woTS  = np.stack([i[0] for i in peaks_woTS])# if np.array_equal(i[0].shape, (10,10))])
io.imshow(concat_movies(peaks_woTS.sample(1000), nrows=40)[0], cmap='viridis')

# unpack series of lists into array, clear all those empty (because close to
# edge, need to fix this)
spots_ims = np.stack([np.ravel(i[0]) for i in spots_ims])# if np.array_equal(i[0].shape, (10,10))])
# also remove it in peaks DF
#peaks.drop(889, inplace=True)
spots_ims = sklearn.preprocessing.StandardScaler().fit_transform(spots_ims)
# 10% of dimensions, height and width
n_components, h, w = 10, 10, 10
# apply dimensionality reduction
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(spots_ims)
# compute cumulative sum of explained variance ratios
# 2 dimensions gets ~91% var
cum_exp_var = np.cumsum(pca.explained_variance_ratio_)
spots_ims_pca = pca.transform(spots_ims)
# get total intensities per frame to check corr with PCA
ints = [np.sum(i) for i in spots_ims]
# first pca has 0.99 pearson corr with intensity, not very useful; drop it
# Second PCA seems to be uncorrelated
np.corrcoef(ints, spots_ims_pca.T[0])
# get the eigenfaces
eigenfaces = pca.components_.reshape((n_components, h, w))
# number them for plot
n_eigenfaces = np.arange(eigenfaces.shape[0])
plot_gallery(eigenfaces, 2, 5)

# drop first PC
spots_ims_pca = spots_ims_pca.T[1:]
# add dimensions to peaks df
for d in range(n_components-1):
    peaks['pca_{0}'.format(d)] = spots_ims_pca[d]


