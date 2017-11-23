from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn

"""
Attempt to Classify spots using a machine learning
"""

#peaks = peaks.read_csv('../output/111617_TL47PP7smFish.csv')
# get 10by10 box containing candidate spots
def get_bbox(center, im, size=10, return_im=True, pad=2):
    """ get 10 by 10 bounding box from center"""
    x, y = center
    s = int(size/2)
    x, y = int(x), int(y)
    bbox = np.s_[y-s:y+s, x-s:x+s]
    im_bbox = im[bbox]
    if return_im:
        if pad:
            im_bbox = np.pad(im_bbox, pad, 'constant', constant_values=0)
        return im_bbox
    else: return bbox


peaks = pd.read_csv('../output/111617_TL47PP7smFish.csv')
peaks['uid'] = np.arange(len(peaks))
# get only peaks not in TS using unique id
TS_certain = peaks.sort_values('signal', ascending=False).groupby('cellid').apply(lambda x: x.iloc[0])
peaks_woTS = peaks[~(peaks.uid.isin(TS_certain.uid.values))]

# smooth images
ims_smooth = {}
for imname in ims:
    fish_im = ims[imname][:,:,0]
    # Convert image to float
    fish_im = normalize_im(fish_im)
    # substract background with strong gaussian blur
    im_bg = skimage.filters.gaussian(fish_im, sigma=50)
    fish_im -= im_bg
    # smooth with gaussian
    ims_smooth[imname] = skimage.filters.gaussian(fish_im)
peaks_woTS = peaks_woTS.apply(lambda x: [normalize_im(get_bbox(x[['x','y']],
                            ims_smooth[x.imname]))], axis=1)
# take a look at spots
#peaks_woTS  = np.stack([i[0] for i in peaks_woTS])# if np.array_equal(i[0].shape, (10,10))])
fig, ax = plt.subplots(1)
peaks_concat = concat_movies(peaks_woTS, nrows=50)[0]
plt.imshow(peaks_concat, cmap='viridis')

def sel_training(peaks_df, ims_dict, s=10, nrows=50):
    """
    Manual click-selection of training set.
    Use a large screen if number of candidate objects is large!

    Arguments
    ---------
    peaks: dataframe
        df with object coordinates and corresponding image name
        Must contain columns ['x','y','imname']
    ims_dict: dictionary of images
    s: int
        size of window to get from image around object coordinates
    nrows: int
        number of rows of array to display object images

    Returns
    ---------
    sel_bool: boolean array
        Can be used to index original `peaks` dataframe. True for selected ims.
    sel_ims: array
        selected objects in bounding box

    """
    def check_borders(coords, im, s):
        """
        Check if coords are closer than s pixels to borders of im
        Return False if too close, convenient for indexing
        """
        dimx, dimy = im.shape
        return (coords.x>s)&(coords.x<dimx)&(coords.y>s)&(coords.y<dimy)

    # create id first keep track of original rows
    peaks = peaks_df.copy()
    peaks['uid'] = np.arange(len(peaks))
    # clear peaks too close to image border in copy of dataframe
    not_inborder = peaks.apply(lambda x: check_borders(x[['x','y']],
                                            ims_dict[x.imname], s), axis=1)
    peaks_ = peaks[not_inborder]
    # get s by s squares containing spots
    peaks_ = peaks_.apply(lambda x: [normalize_im(get_bbox(x[['x','y']],
                            ims_dict[x.imname], s))], axis=1)
    # append extra frames if necessary to make square array with nrows
    extra_frames, im_shape = len(peaks_)%nrows, peaks_.iloc[0][0].shape
    if extra_frames > 0:
        add_frames = nrows-extra_frames
        peaks_ = peaks_.append(pd.Series([[np.zeros(im_shape)]\
                                for f in range(add_frames)]))
    # concatenate squares for selection
    peaks_concat = concat_movies(peaks_, nrows=nrows)[0]
    # create s by s squares with labels to track selection and concatenate
    labels = [[np.full(im_shape, l)] for l in np.arange(len(peaks_))]
    labels = concat_movies(labels, nrows=nrows)[0]
    # display for click selection
    fig, ax = plt.subplots(1, figsize=(25.6, 13.6))
    ax.imshow(peaks_concat, cmap='viridis')# array of frames for visual sel
    ax.imshow(labels, alpha=0.0)# overlay array of squares with invisible labels
    # yticks for guidance, take into account padding
    ax.set_yticks(np.arange(14, nrows*14.2, 10))
    plt.tight_layout()
    # get labels by click
    coords = plt.ginput(10000, timeout=0, show_clicks=True)
    plt.close('all')
    if len(coords)>0:
        # filter selected labels
        selected = {labels[int(c1), int(c2)] for (c2, c1) in coords}
        # get boolean array of selected for indexing original df
        sel_bool = peaks.uid.isin(selected)
        # get selected images, without padding. Need to fetch originals again
        peaks_ = peaks[not_inborder]
        peaks_ = peaks_.apply(lambda x: [normalize_im(get_bbox(x[['x','y']],
                        ims_dict[x.imname], s, pad=False))], axis=1)
        sel_ims  = np.stack([i[0] for i in peaks_[sel_bool]])
        return sel_bool, sel_ims
    else: return None, None
# do manual selection of training set
#sel_ind, sel_ims = sel_training(peaks_woTS, ims_smooth)
peaks_woTS['manual_sel'] = sel_ind
peaks_woTS.to_csv('../output/spot_trainingset/111617_TL47PP7smFish_SpotManualSel.csv', index=False)

# unpack series of lists into array, clear all those empty (because close to
# edge, need to fix this)
spots_ims = peaks_woTS.copy()
spots_ims = np.stack([np.ravel(i[0]) for i in spots_ims])# if np.array_equal(i[0].shape, (10,10))])
# also remove it in peaks DF
#peaks.drop(889, inplace=True)
#spots_ims = sklearn.preprocessing.StandardScaler().fit_transform(spots_ims)
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
spots_ims_tsne = TSNE(n_components=2).fit_transform(spots_ims)
# drop first PC
#spots_ims_pca = spots_ims_pca.T[1:]
# add dimensions to peaks df
for d in range(n_components):
    peaks_woTS_df['pca_{0}'.format(d)] = spots_ims_pca.T[d]

