import pandas as pd
import numpy as np
import skimage
from skimage import io
from skimage.external.tifffile import TiffFile
import trackpy as tp
import pymc3 as pm
import bebi103
from bebi103_legacy import ecdf

from collections import defaultdict
from tqdm import tqdm
import os

from im_utils import *

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib


def segment_sel_nuc(im_nuclei, nuc_clf, imname='nuclei'):
    """
    Segment cells/nuclei and manually classify
    
    imname: str
        image name. Because there is only one image, it can be arbitrary
    """
    # mask projected image using adaptive threshold and create markers
    mask_nuclei = mask_image(im_nuclei, min_size=100, block_size=101,
                selem=skimage.morphology.disk(10))
    nuclei_markers = skimage.measure.label(mask_nuclei)

    # Classification of segmented nuclei ======================================
    # get centroids (so all bbox are same size). Dapi is most visible.
    _seg_coords = pd.DataFrame([i.centroid for i in\
        skimage.measure.regionprops(nuclei_markers, im_nuclei)], columns=['y','x'])

    nuc_regionprops = skimage.measure.regionprops(nuclei_markers, im_nuclei)
    _seg_coords = regionprops2df(nuc_regionprops, props=('label', 'centroid'))
    # expand centroid into x, y coords
    _seg_coords[['y','x']] = _seg_coords.centroid.apply(pd.Series)
    # add image name for sel_training func
    _seg_coords['imname'] = imname

    # classify
    _seg_coords, spot_ims = classify_spots_from_df(_seg_coords, clf, {imname: im_nuclei}, s=50)
    labels_to_remove = _seg_coords[_seg_coords.plabel==False].label
    # remove selected markers
    for l in labels_to_remove:
        nuclei_markers[np.where(np.isclose(nuclei_markers,l))] = 0

    return nuclei_markers

def segment_cellfromnuc(im_cells, nuclei_markers):
    """
    Segment cells by reconstructing from nuclei markers using watershed

    Arguments
    ---------
    im_cells: array
        image of cells
    nuclei_markers: array
        integer labeled image of nuclei markers

    Returns
    ---------
    cell_markers, nuclei_markers: array
        only cells with nuclei and nuclei with cells
        cell markers are dilated to keep particles close to edge.
    mask_cells, mask_nuclei: boolean array
        boolean masks
    """

    mask_cells = mask_image(im_cells, min_size=1000, block_size=151,
        selem=skimage.morphology.disk(15))
    # enlarge mask to keep particles close to edge. Doing this before watershed
    # prevents invasion into other cells and is faster, smart
    mask_cells = skimage.morphology.binary_dilation(mask_cells,
                                        selem=skimage.morphology.disk(10))
    cell_markers = skimage.measure.label(mask_cells)
    # watershed transform using nuclei as basins, also removes cells wo nucleus
    cell_markers = skimage.morphology.watershed(cell_markers,
            nuclei_markers, mask=mask_cells)
    # update masks; keeps only cells with nucleus and viceversa
    mask_cells = cell_markers>0
    mask_nuclei = nuclei_markers>0 * mask_cells

    # ensure use of same labels for nuclei
    nuclei_markers  = mask_nuclei * cell_markers

    # correct any invasion into neighboring nuclei?? Unsure how to do this
    # seems to happen rarely, keep an eye out for it

    return cell_markers, nuclei_markers, mask_cells, mask_nuclei

rrdir = '../data/FISH/20171201/'
ims_projected = load_ims(rrdir+'zprojected/', 'tif')
ims_stack = load_ims(rrdir+'zstacks/', 'STK')

# area and intensity limits
maxint_lim, minor_ax_lim, major_ax_lim, area_lim = (0.1,0.99), (15, 500), (20, 500), (500, 5000)
# try it on a sample
#sample = ['666FISHGal10PP7_s10', '666FISHGal10PP7_s9']

cellnums, imnames = [], []
# dataframe for single transcript peaks
peaks = pd.DataFrame()
seg_coords, seg_ims = pd.DataFrame(), ''
for imname in tqdm(ims_stack):
#for imname in tqdm(sample):
    #if imname in imnames: continue
    # get three-channel z-projected image
    print('analyzing {}'.format(imname))
    im = ims_projected[imname]
    # needed because had incomplete dataset
    #try:
    #    im = ims_projected[imname]
    #except KeyError: continue
    # split channels by color
    fish = im[:,:,0] # smFISH channel
    autof = im[:,:,1] # autofluorescent channel for cell segmentation
    dapi = im[:,:,2] # Dapi/nuclear channel
    # get zstack for FISH
    fish_stack = ims_stack[imname]

    # segment cells and nuclei ===========================================
    print('segmenting cells...')

    # get quality controlled nuclei and cell markers
    cell_markers_enlarged, cell_markers, nuclei_markers,
            mask_cells, mask_nuclei = sel_markers_dict[imname]


    # Identify peaks =========================================================
    # identify transcription particles, diameter of 3 works well
    # this params seem to work decently to identify single transcripts
    # Imaging params: LeicaImagingFacility, 100%int 300msExp 0.2uZstack
    #_parts = tp.locate(fish, 3, minmass=45)
    # for 3D, below seems to work well
    print('identifying peaks...')
    #_parts = tp.locate(fish_stack, 9, minmass=1000)
    #_parts['imname'] = imname

    ## Assign transcripts to cells ====================================================
    ## Get cell label
    #_parts['cell_label'] = _parts.apply(lambda coords:\
    #        cell_markers_enlarged[int(coords.y), int(coords.x)], axis=1)
    ## Get nuclear label
    #_parts['nuc_label'] = _parts.apply(lambda coords:\
    #        nuclei_markers[int(coords.y), int(coords.x)], axis=1)
    ## Get total number of cells identified; cells without mRNA matter too
    #_parts['cell_number'] = len(np.unique(cell_markers))

    #peaks = pd.concat((peaks, _parts))

    print('done')

# filter peaks that are not in cells. If label>0, it is inside cell.
peaks = peaks[(peaks.cell_label>0)].reset_index(drop=True)
peaks.to_csv('../output/{}_smFishPeaks3D.csv'.format(rrdir.split('/')[-2]), index=False)

# =============================================================================
# get nuclei centroid dataframe and nuclei markers for segmentation quality
# control =====================================================================
ims_projected_dapi = load_ims(rrdir+'zprojected/', 'tif', channel=2)
seg_coords = pd.DataFrame()
ims_dapi_seg = {}
nuclei_markers_dict = {}
for imname in tqdm(ims_projected_dapi):
    # get three-channel z-projected image
    print('analyzing {}'.format(imname))
    dapi = ims_projected_dapi[imname]
    # segment nuclei ===========================================
    print('making nuclei mask...')
    mask_nuclei = mask_image(dapi, min_size=100, block_size=101,
                selem=skimage.morphology.disk(10))
    nuclei_markers = skimage.measure.label(mask_nuclei)
    print('finding nuclei centers...')
    nuc_centers = tp.locate(dapi, diameter=15, minmass=np.median(dapi)*50,
            separation=15)
    nuc_seeds = np.full_like(dapi, 0)
    for i, (_, r) in enumerate(nuc_centers.iterrows(), 1):
        nuc_seeds[int(r.y), int(r.x)] = i
    print('watershed reconstruction from centers...')
    # remove seeds not in nuclei mask
    nuc_seeds = nuc_seeds * mask_nuclei
    nuclei_markers = skimage.morphology.watershed(nuclei_markers,
                                    nuc_seeds, mask=mask_nuclei)
    # save markers image
    nuclei_markers_dict[imname] = nuclei_markers
    print('making segmentation image with highlighted boundaries...')
    seg_im = dapi.copy()
    seg_im[skimage.segmentation.find_boundaries(nuclei_markers)] = np.max(dapi)
    ims_dapi_seg[imname] = seg_im

    # get centroids (so all bbox are same size)
    nuc_regionprops = skimage.measure.regionprops(nuclei_markers, dapi)
    _seg_coords = regionprops2df(nuc_regionprops, props=('label', 'centroid'))
    # add image name for sel_training func
    _seg_coords['imname'] = imname
    seg_coords = pd.concat((seg_coords, _seg_coords), ignore_index=True)

# expand centroid into x, y coords
seg_coords[['y','x']] = seg_coords.centroid.apply(pd.Series)
# add cell id
seg_coords['cid'] = seg_coords.apply(lambda x: x.imname+'_'+str(x.label), axis=1)
import pickle
with open('../output/nuc_trainingset/20171201_nuclei_segment_centroids_markers_segim.pkl', 'wb') as f:
    pickle.dump(seg_coords, f)
    pickle.dump(nuclei_markers_dict, f)
    pickle.dump(ims_dapi_seg, f)
#seg_coords.to_csv('../output/nuc_trainingset/20171201_nuclei_segment_centroids.csv', index=False)

# =============================================================================
# Manually clean nuclei markers
# =============================================================================
with open('../output/nuc_trainingset/20171201_nuclei_segment_centroids_markers_segim.pkl', 'rb') as f:
    seg_coords = pickle.load(f)
    nuclei_markers_dict = pickle.load(f)
    ims_dapi_seg = pickle.load(f)

# select bad images (usually more good than bad)
_sel_seg, seg_ims, seg_coords = sel_training(seg_coords, ims_dapi_seg,
                    s=50, ncols=50, figsize=(25.6, 13.6), normall=1)
seg_coords['is_nucleus'] = ~sel_seg
plt.imshow(im_block(seg_ims[~sel_seg], 50, norm=0))

ims_proj_autof = load_ims(rrdir+'zprojected/', 'tif', channel=1)
sel_markers_dict = {}
# remove incorrect segmentation instances and make cell markers
for imname, _seg_coords in seg_coords.groupby('imname'):
    nuclei_markers = nuclei_markers_dict[imname].copy()
    cell_markers, nuclei_markers, mask_cells, mask_nuclei =\
            segment_cellfromnuc(ims_proj_autof[imname], nuclei_markers)
    # Remove crap markers. Has to be done after getting cell markers to
    # prevent invasion into non selected cells
    for crap_label in _seg_coords[_seg_coords.is_nucleus==False].label.values:
        cell_markers[np.where(np.isclose(cell_markers,crap_label))] = 0
        nuclei_markers[np.where(np.isclose(nuclei_markers,crap_label ))] = 0
    sel_markers_dict[imname] = cell_markers, nuclei_markers

    # make a segmentation image with highlighted boundaries
    seg_im = ims_projected_dapi[imname].copy()
    seg_im[skimage.segmentation.find_boundaries(cell_markers)] = np.max(seg_im)
    seg_im[skimage.segmentation.find_boundaries(nuclei_markers)] = np.max(seg_im)
    io.imsave('../output/segmentation/{}_segmentation.tif'.format(imname),
                                        skimage.img_as_int(seg_im))

with open('../output/nuc_trainingset/20171201_manualsel_nuclei_segment_centroids_markers_segim.pkl', 'wb') as f:
    pickle.dump(seg_coords, f)
    pickle.dump(sel_markers_dict, f)


# =============================================================================

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
axes[0].imshow(cell_markers, cmap='tab20')
axes[1].imshow(cell_markers-nuclei_markers, cmap='tab20')
axes[2].imshow(nuclei_markers, cmap='tab20')
axes[3].imshow(ims_projected_dapi[imname])

#seg_coords.to_csv('../output/nuc_trainingset/20171201_nuclei_segment_centroids_manualclass_firstpass.csv', index=False)
seg_coords = pd.read_csv('../output/nuc_trainingset/20171201_nuclei_segment_centroids_manualclass_firstpass.csv')

plt.imshow(im_block(_seg_ims[seg_coords_bk['manual_sel']==True], 50, norm=0))
plt.imshow(im_block(_seg_ims_mark[seg_coords_bk['manual_sel']==True], 50, norm=1))
sel_ind = _seg_coords[~_sel_seg].index.values
seg_coords_bk['manual_sel'] = seg_coords_bk.index.isin(sel_ind)

#        # For visualization only
#        blurred = skimage.filters.gaussian(fish)
#        # change intensity maximum for visualization; otherwise TS blobls overwhelm
#        max_viz = 2*np.median(fish)
#        fish_viz = np.clip(fish, 0, max_viz)
#        # or this is also good for viz
#        fish_viz2 = skimage.exposure.equalize_hist(blurred, mask=mask_cells)
#        im_viz = np.dstack((fish_viz, dapi))
#        # best viz, just use 'gist_stern' colormap
#        io.imshow(fish, cmap='gist_stern')

TSint = peaks[(peaks.nuc_label>0)&(peaks.signal>120)].mass.values
# get intensities of likely single transcripts, only cytoplasmic
mrna_int = peaks[(peaks.nuc_label<1)]
#mrna_int = 100 # median value of mcmc traces with cauchy
# Estimate mass of single transcript
# Cauchy
with pm.Model() as model:
    # Priors
    mu_t = pm.Uniform('mu_t', lower=1, upper=1000)# fluor of single transcript
    beta = bebi103.pm.Jeffreys('beta', lower=1, upper=1000)
    # Likelihood
    mu_obs = pm.Cauchy('mu_obs', alpha=mu_t, beta=beta,
            observed=mrna_int.mass.values)
    trace = pm.sample(draws=2000, tune=2000, init='advi+adapt_diag', njobs=4)
df_mcmc_mrna = bebi103.pm.trace_to_dataframe(trace)
df_mcmc_mrna.to_csv('../output/111617_TL47PP7smFish_singlemRNAFluorMCMC.csv', index=False)
corner.corner((df_mcmc_mrna[['mu_t','beta']]))

# Normal
with pm.Model() as model:
    # Priors
    mu_t = pm.Uniform('mu_t', lower=1, upper=1000)# fluor of single transcript
    sigma_t = bebi103.pm.Jeffreys('sigma_t', lower=1, upper=1000)
    # Likelihood
    mu_obs = pm.Normal('mu_obs', mu=mu_t, sd=sigma_t, observed=mrna_int)
    trace = pm.sample(draws=2000, tune=2000, init='advi+adapt_diag', njobs=4)
df_mcmc_mrna = bebi103.pm.trace_to_dataframe(trace)
corner.corner((df_mcmc_mrna[['mu_t','sigma_t']]))


with pm.Model() as model:
    # Priors
    mu = pm.Uniform('mu', 0, 600)
    sigma = bebi103.pm.Jeffreys('sigma', 0.1, 1000)
    sigma_bad = bebi103.pm.Jeffreys('sigma_bad', sigma, 1000)
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
w = np.median(df_mcmc[cols].values, axis=0)
plt.figure()
corner.corner(df_mcmc[['mu', 'sigma', 'sigma_bad']])
# smaller valued intensities have higher prob of being bad
plt.scatter(w, mrna_int, s=10, alpha=0.2)
# add bad prob and filter
mrna_int['w_bad'] = w
mrna_int_good = mrna_int[(mrna_int['w_bad']<0.2)]

peaks_lbl = pd.read_csv('../output/spot_trainingset/111617_TL47PP7smFish_SpotManualSel.csv')
mrnaint = peaks_lbl[(peaks_lbl.manual_sel=='mrna')&(peaks_lbl.mass<400)].mass.values
TSint = peaks_lbl[(peaks_lbl.manual_sel=='TS')].mass.values

# Estimate number of transcripts per TS
# don't know how to do this bit...
ind = np.arange(len(TS_int))
with pm.Model() as model_TS:
    # Priors
    mu_t = pm.Uniform('mu_t', lower=1, upper=1000)
    sigma_t = bebi103.pm.Jeffreys('sigma_t', lower=1, upper=1000)
    mu_obs = pm.Normal('mu_obs', mu=mu_t, sd=sigma_t, observed=mrnaint)
    alpha = pm.Uniform('alpha', lower=1, upper=1000, shape=len(TSint))
    sigma_a = bebi103.pm.Jeffreys('sigma_a', lower=1, upper=1000, shape=len(TSint))
    # Likelihood
    a_obs = pm.Normal('a_obs', mu=alpha[ind]*mu_t, sd=sigma_a, observed=TSint, shape=len(TSint))
    trace = pm.sample(draws=100, tune=100, init='advi+adapt_diag', njobs=4)

df_mcmc_TS = bebi103.pm.trace_to_dataframe(trace)
TS_mrna = df_mcmc_TS[['alpha__'+str(a) for a in range(0, len(TSint))]].apply(np.median).values
TS_mrna_sigma = df_mcmc_TS[['sigma_a__'+str(a) for a in range(0, len(TSint))]].apply(np.median).values
plt.scatter(*ecdf(TS_mrna), alpha=0.3, s=10)
plt.scatter(*ecdf(TS_mrna_sigma), alpha=0.3, s=10, c='r')

plt.scatter(*ecdf(np.round(TSint/np.median(df_mcmc_mrna['mu_t']))), alpha=0.3, s=10)

# count transcripts by cell
peaks['cellid'] = peaks['imname'] + peaks.cell_label.apply(str)
transcripts_bycell = peaks[(peaks.nuc_label<1)&(peaks.mass>60)].groupby('cellid').x.count().values
plt.scatter(*ecdf(transcripts_bycell), alpha=0.3, s=8)
sns.distplot(transcripts_bycell, bins=21)
# get nuclear peaks
peaks_nuc = peaks[(peaks.nuc_label>0)]
plt.scatter(*ecdf(np.log(peaks[(peaks.nuc_label<1)&(peaks.mass>60)].mass)), alpha=0.3, s=8)

plt.scatter(*ecdf(peaks[(peaks.nuc_label>0)].mass), label='nuclear', alpha=0.3, s=8)
plt.scatter(*ecdf(peaks[(peaks.nuc_label<1)].mass), label='cytoplasmic', alpha=0.3, s=8)
plt.legend()
plt.scatter(*ecdf(np.log(peaks[(peaks.nuc_label>0)].mass)), label='nuclear', alpha=0.3, s=8)
plt.scatter(*ecdf(np.log(peaks[(peaks.nuc_label<1)].mass)), label='cytoplasmic', alpha=0.3, s=8)
plt.legend()
sns.kdeplot(np.log(peaks[(peaks.nuc_label>0)].mass), label='nuclear')
sns.kdeplot(np.log(peaks[(peaks.nuc_label<1)].mass), label='cytoplasmic')
plt.legend()

mask_cell_enlarged = cell_markers_enlarged>0
TS_certain = peaks.sort_values('signal', ascending=False).groupby('cellid').apply(lambda x: x.iloc[0])
# take a look only at spots, these are pretty good
ts_certain_im = TS_certain.apply(lambda x: [get_bbox(x[['x','y']],
                            ims[x.imname][:,:,0])], axis=1)
plt.figure()
io.imshow(concat_movies(ts_certain_im, nrows=17)[0], cmap='gist_stern')


fig, axes = plt.subplots(3, sharex=True, sharey=True)
tp.annotate(peaks[peaks.imname==imname], fish, ax=axes[0], imshow_style={'cmap':'gist_stern'})
#tp.annotate(peaks[(peaks.imname==imname)&((peaks.nuc_label>0)|(peaks.signal>120))], fish, ax=axes[0], imshow_style={'cmap':'gist_stern'})
axes[0].imshow(mask_nuclei.astype(int)+mask_cell_enlarged, alpha=0.4, cmap='Accent_r')
axes[1].imshow(fish,cmap='gist_stern')
axes[2].imshow(dapi)

# Test TL GaussianMaskFit2 with easy spot
test_spot = peaks[peaks.imname==imname].iloc[48]
# coordinates of cell with easy spot, obtained with zoom2roi
test_cell_coords = (slice(106, 359, None), slice(768, 1051, None))
test_cell = fish[test_cell_coords]
# vars for TL GaussianMaskFit2
s = 1.5
im = test_cell.copy()
# transform coords to cropped im coords
coo = np.sum((test_spot[['x','y']], (-768, -106)), axis=0)
# make sure everything is fine
plt.imshow(im, cmap='gist_stern')
plt.scatter(*coo, s=50, c='b')
