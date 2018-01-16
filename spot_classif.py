from matplotlib import pyplot as plt 
from im_utils import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn
import pickle

"""
Classify spots using SVMs
"""

# load z-projected movies
ims = {}
ddir = '../data/FISH/111617_TL47WT_GAL10PP7Cy5_100%int300ms02uZslice/'
for fname in tqdm(os.listdir(ddir)):
    if 'tif' in fname:
        ims[fname.split('.')[0]] = io.imread(ddir + fname)
    else: next

# get smFISH image, substract gaussian background and smooth
ims_smooth = {}
for imname in ims:
    fish_im = ims[imname][:,:,0].copy()
    # Convert image to float
    fish_im = normalize_im(fish_im)
    # substract background with strong gaussian blur
    im_bg = skimage.filters.gaussian(fish_im, sigma=50)
    fish_im -= im_bg
    # smooth with gaussian
    fish_im = skimage.filters.gaussian(fish_im, sigma=1)
    ims_smooth[imname] = fish_im
peaks_woTS = peaks_woTS.apply(lambda x: [normalize_im(get_bbox(x[['x','y']],
                            ims_smooth[x.imname]))], axis=1)

peaks = pd.read_csv('../output/111617_TL47PP7smFish.csv')

peaks = pd.read_csv('../output/spot_trainingset/111617_TL47PP7smFish_SpotManualSel.csv')
# clear spots too close to image borders
not_inborder = peaks.apply(lambda x: check_borders(x[['x','y']],
                                        ims_smooth[x.imname], 10), axis=1)
peaks = peaks[not_inborder]
# manual selection of training set from pre-selected candidate peaks
mrna_sorted = peaks[peaks.manual_sel=='crap'].sort_values('signal')
sel_ind, ims_training = sel_training(mrna_sorted, ims_smooth, nrows=32, scale=0.5, s=25, cmap='gist_stern')

# get only peaks not in TS using unique id
TS_certain = peaks.sort_values('signal', ascending=False).groupby('cellid').apply(lambda x: x.iloc[0])
peaks_woTS = peaks[~(peaks.uid.isin(TS_certain.uid.values))]


peaks['uid'] = np.arange(len(peaks))
peaks_ts = peaks[(peaks.manual_sel==1)&(peaks.signal>120)]
sel_ind_ts, ims_training_ts = sel_training(peaks_ts, ims_smooth, s=25, nrows=10, cmap='gist_stern')
peaks_ts['TS'] = np.logical_not(sel_ind_ts)
peaks.loc[peaks.uid.isin(peaks_ts[peaks_ts.TS==1].uid), 'manual_sel'] = 'TS'
peaks.loc[peaks.manual_sel==1, 'manual_sel'] = 'mrna'
peaks.loc[peaks.manual_sel==0, 'manual_sel'] = 'crap'

peaks_r = peaks[peaks.manual_sel=='crap'].sort_values('mass')

ind_r = []
ims_r = []
step = 400
for n in np.arange(step, len(peaks_r), step):
    if n > len(peaks_r) - step:
        ind_, ims_= sel_training(peaks_r[n:], ims_smooth, s=10, nrows=15, cmap='viridis')
    else:
        ind_, ims_= sel_training(peaks_r[n-step:n], ims_smooth, s=10, nrows=15, cmap='viridis')
    ind_r.append(ind_)
    ims_r.append(ims_)

ind_, ims_= sel_training(peaks_r[0:step], ims_smooth, s=5, nrows=15, cmap='gist_stern')

#ims_r_all = mrna.apply(lambda x: [get_bbox(x[['x','y']],
#                ims_smooth[x.imname], 10, pad=False)], axis=1)
ims_r_all  = np.stack([i[0] for i in ims_r_all ])
ind_r_all = np.concatenate(ind_)
ims_r_all = np.concatenate(ims_)

#update peaks after 4 revs
# reset mrna selection
peaks['manual_sel'] = peaks.manual_sel.str.replace('mrna','crap')
peaks.loc[peaks.uid.isin(mrna4[mrna4.rev3==True].uid), 'manual_sel'] = 'mrna'

with open('../output/spot_trainingset/lauren_sel_mrna.pkl', 'wb') as f:
    pickle.dump(ind_lauren_all, f)
    pickle.dump(ims_lauren_all, f)

io.imshow(im_block(ims_lauren_all[ind_lauren_all], 20, norm=1), cmap='viridis')
io.imshow(im_block(ims_lauren_all[~ind_lauren_all], 50, norm=1), cmap='viridis')

io.imshow(im_block(ims_training[peaks.manual_sel=='mrna'], 50, norm=1), cmap='viridis')
plt.figure()
io.imshow(im_block(ims_training[(peaks.manual_sel=='TS')], 15, norm=1), cmap='viridis')

fig, axes = plt.subplots(2, sharex=True, sharey=True)
axes[0].imshow(im_block(ims_training, 75, norm=1), cmap='viridis')
axes[1].imshow(np.clip(im_block(ims_training, 75, norm=0), 0, 0.05), cmap='viridis')

io.imshow(np.clip(im_block(ims_training[peaks.manual_sel=='mrna'], 25, norm=0), 0, 0.025), cmap='viridis')
plt.figure()
io.imshow(np.clip(im_block(ims_training[peaks.manual_sel=='TS'], 15, norm=0), 0, 0.85), cmap='viridis')

io.imshow(np.clip(im_block(spot_ims[spot_labels=='mrna'], 50, norm=0), 0, 0.025), cmap='viridis')

io.imshow(np.clip(im_block(spot_ims[spot_labels=='crap'], 75, norm=0), 0, 0.025), cmap='viridis')
io.imshow(np.clip(im_block(spot_ims[spot_labels=='mrna'], 50, norm=0), 0, 0.025), cmap='viridis')
io.imshow(np.clip(im_block(spot_ims[spot_labels=='TS'], 15, norm=0), 0, 0.025), cmap='viridis')
plt.figure()
io.imshow(im_block(spot_ims[spot_labels=='mrna'], 50, norm=1), cmap='viridis')
plt.figure()
io.imshow(im_block(spot_ims[spot_labels=='TS'], 15, norm=1), cmap='viridis')


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
plot_ecdf(peaks[(peaks.manual_sel=='crap')].mass.values, 0)
plot_ecdf(peaks[(peaks.manual_sel=='TS')].mass.values, 0)
plot_ecdf(peaks[(peaks.manual_sel==1)&(peaks.TS==1)].mass.values, 0)
plot_ecdf(peaks[(peaks.manual_sel==1)&(peaks.signal>120)].mass.values, 0)
sns.distplot(peaks[(peaks.manual_sel==1)].mass.values)
# There are clearly two distributions here,
# probably single transcripts and more than one



#peaks_woTS.to_csv('../output/spot_trainingset/111617_TL47PP7smFish_SpotManualSel.csv', index=False)
peaks = pd.read_csv('../output/spot_trainingset/111617_TL47PP7smFish_SpotManualSel.csv')
peaks_sel = peaks_woTS[peaks_woTS.manual_sel==0]

# pickle all images and labels
with open('../output/spot_trainingset/111617_TL47PP7smFish_Training_Labels_ImsSmoothMinusBkUnnorm.pkl', 'wb') as f:
    pickle.dump(peaks.manual_sel.values, f)
    pickle.dump(ims_training, f)

# unpack series of lists into array, clear all those empty (because close to
# edge, need to fix this)
#spot_ims = peaks_woTS.copy()
not_inborder = peaks_woTS.apply(lambda x: check_borders(x[['x','y']],
                                            ims_smooth[x.imname], 10), axis=1)
peaks_woTS = peaks_woTS[not_inborder]
peaks_woTS = peaks_woTS.apply(lambda x: [get_bbox(x[['x','y']],
                            ims_smooth[x.imname], pad=0)], axis=1)
ims_training = np.stack([i[0] for i in peaks_woTS])

#############################################################################
# PCA
#############################################################################
sel_ind = peaks.manual_sel=='crap'
spot_ims = ims_training.copy()


spot_ims = np.stack([np.ravel(i) for i in spot_ims])
# also remove it in peaks DF
#peaks.drop(889, inplace=True)
#spot_ims = sklearn.preprocessing.StandardScaler().fit_transform(spot_ims)
# 10% of dimensions, height and width
n_components, h, w = 10, 9, 9
# apply dimensionality reduction
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(spot_ims)
# compute cumulative sum of explained variance ratios
# 2 dimensions gets ~91% var
cum_exp_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cum_exp_var)
spot_ims_pca = pca.transform(spot_ims)
# get total intensities per frame to check corr with PCA
ints = [np.sum(i) for i in spot_ims]
# first pca has 0.99 pearson corr with intensity, not very useful; drop it
# Second PCA seems to be uncorrelated
np.corrcoef(ints, spot_ims_pca.T[0])
# get the eigenfaces
eigenfaces = pca.components_.reshape((n_components, h, w))
# number them for plot
n_eigenfaces = np.arange(eigenfaces.shape[0])
plot_gallery(eigenfaces, 2, 5)
# look at distributions of PC
fig, axes = plt.subplots(3, 3)
axes = iter(axes.ravel())
for i, ax in enumerate(axes):
    ax.plot(*ecdf(spot_ims_pca.T[i][sel_ind], conventional=True), label='sel')
    ax.plot(*ecdf(spot_ims_pca.T[i][~sel_ind], conventional=True), label='not_sel')
    plt.legend()
# Spots may be linearly separable, PCs 2 and 3 seem specially useful
fig, axes = plt.subplots(1)
axes.scatter(spot_ims_pca.T[0][~sel_ind], spot_ims_pca.T[2][~sel_ind], c='b', label='not sel', alpha=0.3)
axes.scatter(spot_ims_pca.T[0][sel_ind], spot_ims_pca.T[2][sel_ind], c='r', label='sel', alpha=0.3)
plt.legend()


spot_ims_tsne = TSNE(n_components=10).fit_transform(spot_ims)
fig, axes = plt.subplots(1)
axes.scatter(spot_ims_tsne.T[2][~sel_ind], spot_ims_tsne.T[1][~sel_ind], c='b', label='not sel', alpha=0.3)
axes.scatter(spot_ims_tsne.T[2][sel_ind], spot_ims_tsne.T[1][sel_ind], c='r', label='sel', alpha=0.3)
plt.legend()


################################################################################
# Try SVM classification
################################################################################
from sklearn import model_selection
from sklearn import metrics
from sklearn import svm
from sklearn.utils.extmath import fast_dot

with open('../output/spot_trainingset/111617_TL47PP7smFish_Training_ImsNormNotSmoothSize9.pkl', 'rb') as f:
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
C_range = np.logspace(-6, 1, 10)
gamma_range = np.logspace(-6, 1, 10)
param_grid = dict(gamma=gamma_range, C=C_range)
clf = model_selection.GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'),
                                   param_grid)
clf = clf.fit(spot_train, labels_train)
# one class SVM
#clf = model_selection.GridSearchCV(svm.OneClassSVM(kernel='rbf', class_weight='balanced'),
#                                   param_grid)
#clf = clf.fit(spot_train)

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
# save classifier
with open('../output/spot_trainingset/clf_beta1_size9NormedNotSmooth.pkl', 'wb') as f:
    pickle.dump(clf, f)


# get misclassified
miscl = labels_pred!=labels_test
spot_miscl = normalize(spot_test[miscl])
label_true_miscl = labels_test[miscl]
label_pred_miscl = labels_pred[miscl]
labels_cl = ['True:{0}'.format(str(l1),str(l2)) for (l1, l2) in zip(label_true_miscl, label_pred_miscl)]
plt.figure()
plot_gallery(spot_miscl, 1, 4, reshape=(9,9), titles=labels_cl)

# get indices to randomly look at our predictions
rows, cols = 10, 29
r_ind = np.random.choice(spot_test.shape[0], size=rows*cols, replace=False)

# get pictures and respective predictions to look at
s_photos = spot_test[r_ind]
labels_pred_sample = labels_pred[r_ind]

# plot predictions
plot_gallery(s_photos, 10, 10, titles=labels_pred_sample, n_row=rows, n_col=cols, fig_title="Predictions")

# Classify PP7 spots ========================================================

# get a sample of brightest spots per trajectory
#pp7['pid'] = pp7.apply(lambda x: str(x.particle)+x.imname, axis=1)
#pp7_brightest = pp7.sort_values('mass').drop_duplicates('pid', keep='last')
#pp7_sample = pp7_brightest.sample(500, random_state=42)
#not_inborder = pp7_sample.apply(lambda x: check_borders(x[['x','y']],
#                                            movs[x.imname][x.frame], s), axis=1)
#pp7_sample = pp7_sample.loc[not_inborder]
## found spots with low minmass (stored in _parts) to use as crap class for SVM
#crap = _parts[_parts<500].sample(1000, random_state=42)
## label images of sample
#io.imshow(im_block(spot_ims[spot_labels], 3, norm=1), cmap='viridis')
#plt.figure()
#io.imshow(im_block(spot_ims[~spot_labels], 20, norm=1), cmap='viridis')
#real_parts_pid = pp7_sample[spot_labels].pid.values
#real_parts = pp7[pp7.pid.isin(real_parts_pid)]
#sel = _spot_ims[:150]
#sel_crap = spot_ims[~spot_labels]
#io.imshow(im_block(sel_crap, 20, norm=1), cmap='viridis')
#spot_ims = np.concatenate((sel, sel_crap))
#spot_labels = np.concatenate((np.full(len(sel), 1), np.full(len(sel_crap),0)))
##shuffle
#ind = np.arange(0, len(spot_ims))
#np.random.shuffle(ind)
#spot_ims, spot_labels = spot_ims[ind], spot_labels[ind]
#spot_labels = spot_labels>0
#spot_ims = normalize(spot_ims)
#io.imshow(im_block(spot_ims[spot_labels>0], 20, norm=0), cmap='viridis')

# train classifier, using code above in the meantime
# save classifier
#with open('../output/pp7/clf_beta0_pp7spots.pkl', 'wb') as f:
#    pickle.dump(clf, f)

# Classify PP7 images ========================================================
# load classifier
with open('../output/pp7/clf_beta0_pp7spots.pkl', 'rb') as f:
    clf = pickle.load(f)
pp7 = pd.read_pickle('../output/pp7/nuclei_peaks.p')
# classify!
#pp7_clf = classify_spots_from_df(pp7, clf, movs_smooth, 10, movie=True)
# Need to normalize: it's structure, not intensity that matters
pp7_labels_pred, pp7_ims = classify_spots_from_df(pp7, clf, movs, 9,
        movie=True, norm=True)
# check them out
plt.imshow(im_block(pp7_ims[pp7_labels_pred.plabel==True], 100), cmap='gist_stern')
# seems alright, just get those trajectories with at least two good spots
svm_good_parts = pp7_labels_pred[pp7_labels_pred.plabel==True]
svm_good_parts['pid'] = svm_good_parts.apply(lambda x: str(x.particle)+x.imname, axis=1)
svm_good_parts = filter_parts(svm_good_parts, thresh=2)
# save
with open('../output/pp7/pp7spots_SVMfiltered_20171219.pkl', 'wb') as f:
    pickle.dump(svm_good_parts, f)

# manually select cells to trash
movs_proj = {}
for k in movs:
    movs_proj[k] = z_project(movs[k])

# remove unloaded movies from dataframe
pp7 = pp7[pp7.imname.isin(movs_proj.keys())]

pp7['cid'] = pp7.apply(lambda x: str(x.label)+x.imname, axis=1)
pp7_ucid = pp7.sort_values('mass', ascending=False).drop_duplicates('cid')
xx = pp7_ucid.apply(lambda x: (x.bbox[0] + x.bbox[2])/2, axis=1)
yy = pp7_ucid.apply(lambda x: (x.bbox[1] + x.bbox[3])/2, axis=1)
pp7_ucid['x'] = yy
pp7_ucid['y'] = xx

spot_labels, spot_ims, filt_pp7df = sel_training(pp7_ucid, movs_proj, movie=0,
        scale=1, ncols=20, mark_center=0, s=30, normall=0)
sel_cid = filt_pp7df[spot_labels].cid
nuclei_peaks = pp7[pp7.cid.isin(sel_cid)]

# make movie of all for reference selection
nuc_movs = []
norm=True#normalize each frame to improve viz
for (name, label), nuc in pp7[pp7.cid.isin(sel_cid)].groupby(['imname', 'label'], sort=False):
    # get whole movie from original
    b = nuc.bbox.iloc[0] # bounding box coordinates are the same for all frames
    bx, by = (slice(b[0], b[2]), slice(b[1], b[3]))
    nuc_mov = movs[name][:, bx, by]
    nuc_trackmov = tracking_movie(nuc_mov, nuc, x='bbx', y='bby')
    nuc_movs.append(nuc_trackmov)
nuc_movs = [skimage.img_as_uint(normalize(nuc_movs[i])) for i in range(len(nuc_movs))]
globmov = concat_movies(nuc_movs, nrows=12)
io.imsave('../output/pp7/tracking_movs_classif/allmovies_ref.tif'.format(m), globmov)

# manually select particles to trash
s=30
pp7_upid = pp7.sort_values('mass', ascending=False).drop_duplicates('pid')
spot_labels, spot_ims, filt_pp7df = sel_training(pp7_upid, movs_proj, movie=0,
        scale=1, ncols=20, mark_center=0, s=s, normall=0)
sel_pids = pp7_upid[spot_labels].pid
with open('../output/pp7/pp7spots_handselected_20180105.pkl', 'wb') as f:
    pickle.dump(pp7[pp7.pid.isin(sel_pids)], f)
