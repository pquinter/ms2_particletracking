from matplotlib import pyplot as plt import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn
import pickle

"""
Attempt to Classify spots using a machine learning
"""

peaks = pd.read_csv('../output/111617_TL47PP7smFish.csv')
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


peaks['uid'] = np.arange(len(peaks))
# get only peaks not in TS using unique id
TS_certain = peaks.sort_values('signal', ascending=False).groupby('cellid').apply(lambda x: x.iloc[0])
peaks_woTS = peaks[~(peaks.uid.isin(TS_certain.uid.values))]

# smooth images and substract gaussian background
ims_smooth = {}
for imname in ims:
    fish_im = ims[imname][:,:,0]
    # Convert image to float
    fish_im = normalize_im(fish_im)
    # substract background with strong gaussian blur
    im_bg = skimage.filters.gaussian(fish_im, sigma=50)
    fish_im -= im_bg
    # smooth with gaussian
    #ims_smooth[imname] = skimage.filters.gaussian(fish_im, sigma=1)
    ims_smooth[imname] = fish_im
peaks_woTS = peaks_woTS.apply(lambda x: [normalize_im(get_bbox(x[['x','y']],
                            ims_smooth[x.imname]))], axis=1)
# take a look at spots
#peaks_woTS  = np.stack([i[0] for i in peaks_woTS])# if np.array_equal(i[0].shape, (10,10))])
fig, ax = plt.subplots(1)
peaks_concat = concat_movies(peaks_woTS, nrows=50)[0]
plt.imshow(peaks_concat, cmap='viridis')

def sel_training(peaks_df, ims_dict, s=10, nrows=50, cmap='viridis'):
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
    all_ims: array
        Screened objects. To get selected, index as ims[sel_bool]

    """
    def check_borders(coords, im, s):
        """
        Check if coords are closer than s pixels to borders of im
        Return False if too close, convenient for indexing
        """
        dimx, dimy = im.shape
        return (coords.x>s)&(coords.x+s<dimx)&(coords.y>s)&(coords.y+s<dimy)

    # create id first keep track of original rows
    peaks = peaks_df.copy()
    # clear peaks too close to image border
    not_inborder = peaks.apply(lambda x: check_borders(x[['x','y']],
                                            ims_dict[x.imname], s), axis=1)
    peaks = peaks.loc[not_inborder]
    peaks['uid'] = np.arange(len(peaks))
    # get s by s squares containing spots
    peaks_ims = peaks.apply(lambda x: [normalize_im(get_bbox(x[['x','y']],
                            ims_dict[x.imname], s))], axis=1)
    # append extra frames if necessary to make square array with nrows
    extra_frames, im_shape = len(peaks_ims)%nrows, peaks_ims.iloc[0][0].shape
    if extra_frames > 0:
        add_frames = nrows-extra_frames
        peaks_ims = peaks_ims.append(pd.Series([[np.zeros(im_shape)]\
                                for f in range(add_frames)]))
    # concatenate squares for selection
    peaks_imsconcat = concat_movies(peaks_ims, nrows=nrows)[0]
    # create s by s squares with labels to track selection and concatenate
    labels = [[np.full(im_shape, l)] for l in range(len(peaks_ims))]
    labels = concat_movies(labels, nrows=nrows)[0]
    # display for click selection
    fig, ax = plt.subplots(1, figsize=(25.6, 13.6))
    ax.set_title('click to select; ctrl+click to undo last click; alt+click to finish')
    ax.imshow(peaks_imsconcat, cmap=cmap)# array of frames for visual sel
    ax.imshow(labels, alpha=0.0)# overlay array of squares with invisible labels
    # yticks for guidance, take into account padding
    ax.set_yticks(np.arange(s+4, nrows*1.1*s+4, 10))
    plt.tight_layout()
    # get labels by click
    coords = plt.ginput(10000, timeout=0, show_clicks=True)
    plt.close('all')
    if len(coords)>0:
        # filter selected labels
        selected = {labels[int(c1), int(c2)] for (c2, c1) in coords}
    else: selected = []
    # get boolean array of selected for indexing original df
    sel_bool = peaks.uid.isin(selected).values
    # get selected images, without padding nor normalizing. Need to fetch originals again
    peaks_ims = peaks.apply(lambda x: [get_bbox(x[['x','y']],
                    ims_dict[x.imname], s, pad=False)], axis=1)
    all_ims  = np.stack([i[0] for i in peaks_ims])
    return sel_bool, all_ims
# do manual selection of training set from pre-selected candidate peaks
peaks = pd.read_csv('../output/111617_TL47PP7smFish.csv')
#sel_ind, ims_training = sel_training(peaks, ims_smooth, nrows=50)

peaks = pd.read_csv('../output/spot_trainingset/111617_TL47PP7smFish_SpotManualSel.csv')
peaks['uid'] = np.arange(len(peaks))
peaks_ts = peaks[(peaks.manual_sel==1)&(peaks.signal>120)]
sel_ind_ts, ims_training_ts = sel_training(peaks_ts, ims_smooth, s=25, nrows=10, cmap='gist_stern')
peaks_ts['TS'] = np.logical_not(sel_ind_ts)
peaks.loc[peaks.uid.isin(peaks_ts[peaks_ts.TS==1].uid), 'manual_sel'] = 'TS'
peaks.loc[peaks.manual_sel==1, 'manual_sel'] = 'mrna'
peaks.loc[peaks.manual_sel==0, 'manual_sel'] = 'crap'

mrna = peaks[(peaks.manual_sel=='mrna')]
not_inborder = mrna.apply(lambda x: check_borders(x[['x','y']],
                                        ims_smooth[x.imname], 10), axis=1)
mrna = mrna.loc[not_inborder]
sel_ind_r, ims_training_r = sel_training(mrna, ims_smooth, s=10, nrows=30, cmap='viridis')
peaks.loc[peaks.uid.isin(crap[sel_ind_r].uid), 'manual_sel'] = 'mrna'
io.imshow(im_block(ims_training[(peaks.manual_sel=='crap')], 50, norm=1), cmap='viridis')
io.imshow(im_block(ims_training_r[sel_ind_r], 10, norm=1), cmap='viridis')
io.imshow(np.hstack(normalize(ims_training_r[sel_ind_r])), cmap='viridis')

# clear peaks near border, done inside sel_training
peaks = peaks[peaks.apply(lambda x: check_borders(x[['x','y']],
                                        ims_dict[x.imname], s), axis=1)]
# now can add label
peaks['manual_sel'] = sel_ind
peaks.to_csv('../output/spot_trainingset/111617_TL47PP7smFish_SpotManualSel.csv', index=False)
# look at distribution of intensities
plt.figure()
plot_ecdf(peaks[(peaks.manual_sel=='mrna')].mass.values, 0)
plot_ecdf(peaks[(peaks.manual_sel=='TS')].mass.values, 0)
plot_ecdf(peaks[(peaks.manual_sel==1)&(peaks.TS==1)].mass.values, 0)
plot_ecdf(peaks[(peaks.manual_sel==1)&(peaks.signal>120)].mass.values, 0)
sns.distplot(peaks[(peaks.manual_sel==1)].mass.values)
# There are clearly two distributions here,
# probably single transcripts and more than one



#peaks_woTS.to_csv('../output/spot_trainingset/111617_TL47PP7smFish_SpotManualSel.csv', index=False)
peaks_woTS = pd.read_csv('../output/spot_trainingset/111617_TL47PP7smFish_SpotManualSel.csv')
peaks_sel = peaks_woTS[peaks_woTS.manual_sel==0]

# pickle all images and labels
with open('../output/spot_trainingset/111617_TL47PP7smFish_Training_Labels_ImsSmoothMinusBkUnnorm.pkl', 'wb') as f:
    pickle.dump(peaks.manual_sel.values, f)
    pickle.dump(ims_training, f)

# unpack series of lists into array, clear all those empty (because close to
# edge, need to fix this)
#spots_ims = peaks_woTS.copy()
not_inborder = peaks_woTS.apply(lambda x: check_borders(x[['x','y']],
                                            ims_smooth[x.imname], 10), axis=1)
peaks_woTS = peaks_woTS[not_inborder]
peaks_woTS = peaks_woTS.apply(lambda x: [get_bbox(x[['x','y']],
                            ims_smooth[x.imname], pad=0)], axis=1)
ims_training = np.stack([i[0] for i in peaks_woTS])

#############################################################################
# PCA
#############################################################################
sel_ind = peaks.manual_sel=='mrna'
spots_ims = ims_training#[peaks.manual_sel=='mrna']
spots_ims = np.stack([np.ravel(i) for i in spots_ims])
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
plt.plot(cum_exp_var)
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
# look at distributions of PC
fig, axes = plt.subplots(3, 3)
axes = iter(axes.ravel())
for i, ax in enumerate(axes):
    ax.plot(*ecdf(spots_ims_pca.T[i][sel_ind], conventional=True), label='sel')
    ax.plot(*ecdf(spots_ims_pca.T[i][~sel_ind], conventional=True), label='not_sel')
    plt.legend()
# Spots may be linearly separable, PCs 2 and 3 seem specially useful
fig, axes = plt.subplots(1)
axes.scatter(spots_ims_pca.T[1][~sel_ind], spots_ims_pca.T[2][~sel_ind], c='b', label='not sel', alpha=0.3)
axes.scatter(spots_ims_pca.T[1][sel_ind], spots_ims_pca.T[2][sel_ind], c='r', label='sel', alpha=0.3)
plt.legend()


spots_ims_tsne = TSNE(n_components=10).fit_transform(spots_ims)
fig, axes = plt.subplots(1)
axes.scatter(spots_ims_tsne.T[2][~sel_ind], spots_ims_tsne.T[1][~sel_ind], c='b', label='not sel', alpha=0.3)
axes.scatter(spots_ims_tsne.T[2][sel_ind], spots_ims_tsne.T[1][sel_ind], c='r', label='sel', alpha=0.3)
plt.legend()


################################################################################
# Try SVM classification
################################################################################
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.utils.extmath import fast_dot

with open('../output/spot_trainingset/111617_TL47PP7smFish_Training_Labels_ImsSmoothMinusBkUnnorm.pkl', 'rb') as f:
    spot_labels = pickle.load(f)
    spot_ims = pickle.load(f)
# Each image must not be normalized independently! Int is important for finding TS
# Ravel images.
spot_ims = np.stack([np.ravel(i) for i in spot_ims])
# split into a training and testing set
spot_train, spot_test, labels_train, labels_test = \
    model_selection.train_test_split(spot_ims, spot_labels, test_size=0.25,
                                     random_state=42)

#SVM with an RBF kernel, which takes two parameters, C
# Find best pair of parameters and train
C_range = np.logspace(-1, 5, 4)
gamma_range = np.logspace(-3, 0, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
clf = model_selection.GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                                   param_grid)
clf = clf.fit(spot_train, labels_train)
scores = clf.cv_results_['mean_test_score'].reshape(len(C_range), 
                                                    len(gamma_range))

plt.contourf(scores, cmap=plt.cm.viridis)
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.ylabel(r'$C$')
plt.xlabel(r'$\gamma$')
print('The best parameters are {0} with a score of {1:.2f}'.format(
        clf.best_params_, clf.best_score_))
np.random.seed(10)
# predict labels and get report
labels_pred = clf.predict(spot_test)
print(metrics.classification_report(labels_test, labels_pred))
# It's not bad
"""
The best parameters are {'C': 100000.0, 'gamma': 1.0} with a score of 0.93
             precision    recall  f1-score   support

      False       0.95      0.92      0.94       495
       True       0.91      0.94      0.93       435

avg / total       0.93      0.93      0.93       930
"""
# get misclassified
miscl = labels_pred!=labels_test
spot_miscl = normalize(spot_test[miscl])
label_true_miscl = labels_test[miscl]
label_pred_miscl = labels_pred[miscl]
labels_cl = ['True:{0}; Pred:{1}'.format(str(l1),str(l2)) for (l1, l2) in zip(label_true_miscl, label_pred_miscl)]
plt.figure()
plot_gallery(spot_miscl, labels_cl, 10, 10, 8, 16)

# get indices to randomly look at our predictions
rows, cols = 10, 29
r_ind = np.random.choice(spot_test.shape[0], size=rows*cols, replace=False)

# get pictures and respective predictions to look at
s_photos = spot_test[r_ind]
labels_pred_sample = labels_pred[r_ind]

# plot predictions
plot_gallery(s_photos, labels_pred_sample, 10, 10, n_row=rows, n_col=cols, fig_title="Predictions")
def plot_gallery(images, titles, h, w, n_row=3, n_col=4, fig_title=None):
    """
    Helper function to plot a gallery of portraits
    """
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        ax.set_title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    if fig_title: fig.suptitle(fig_title+'\n', fontsize=20)

def im_block(ims, cols, norm=True):
    if norm: 
        ims = normalize(ims)
    nrows = int(ims.shape[0]/cols)
    xdim, ydim = ims.shape[1:]
    block = []
    for c in np.arange(0, cols*nrows, cols):
        block.append(np.hstack(ims[c:c+cols]))
    block = np.vstack(block)
    return block


