import pandas as pd
import numpy as np
import skimage
from skimage import io
import trackpy as tp

from collections import defaultdict
from tqdm import tqdm
import os

from im_utils import *

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib

# load z-projected movies
ims = {}
ddir = '../data/FISH/111617_TL47WT_GAL10PP7Cy5_100%int300ms02uZslice/'
for fname in tqdm(os.listdir(ddir)):
    if 'tif' in fname:
        ims[fname.split('.')[0]] = io.imread(ddir + fname)
    else: next

# area and intensity limits
maxint_lim, minor_ax_lim, major_ax_lim, area_lim = (0.1,0.99), (15, 500), (20, 500), (500, 5000)

# dataframe for single transcript peaks
peaks = pd.DataFrame()

def segment_cellnuc(im_cells, im_nuclei):
    """
    Segment cells/nuclei
    """
    # mask projected image using adaptive threshold
    mask_cells = mask_image(im_cells, min_size=100, block_size=151,
        selem=skimage.morphology.disk(15))
        #im_thresh=im_cells>skimage.filters.threshold_otsu(im_cells))
    mask_nuclei = mask_image(im_nuclei, min_size=100, block_size=101,
                selem=skimage.morphology.disk(10))

    nuclei_markers = skimage.measure.label(mask_nuclei)
    cell_markers = skimage.measure.label(mask_cells)

    # watershed transform using nuclei as basins, also removes cells wo nucleus
    cell_markers = skimage.morphology.watershed(cell_markers,
            nuclei_markers, mask=mask_cells)
    # ensure use of same labels for nuclei
    nuclei_markers  = mask_nuclei * cell_markers

    # enlarge cell markers to keep particles close to edge
    cell_markers_enlarged = skimage.morphology.dilation(cell_markers,
                selem=skimage.morphology.disk(10))

    return cell_markers_enlarged, cell_markers, nuclei_markers, mask_cells, mask_nuclei

fig, axes = plt.subplots(3, 3)
axes = iter(axes.ravel())

for imname in tqdm(ims):

    # get three-channel image
    print('analyzing {}'.format(imname))
    im = ims[imname]

    # split channels by color
    fish = im[:,:,0] # smFISH channel
    autof = im[:,:,1] # autofluorescent channel for cell segmentation
    dapi = im[:,:,2] # Dapi/nuclear channel

    # For visualization only
    blurred = skimage.filters.gaussian(fish)
    # change intensity maximum for visualization; otherwise TS blobls overwhelm
    max_viz = 2*np.median(fish)
    fish_viz = np.clip(fish, 0, max_viz)
    # or this is also good for viz
    fish_viz2 = skimage.exposure.equalize_hist(blurred, mask=mask_cells)
    im_viz = np.dstack((fish_viz, dapi))

    # segment cells and nuclei ===========================================
    print('segmenting cells...')
    cell_markers_enlarged, cell_markers, nuclei_markers,\
            mask_cells, mask_nuclei = segment_cellnuc(autof, dapi)

    #fig, axes = plt.subplots(2, sharex=True, sharey=True)
    #axes[0].imshow(cell_markers_enlarged)
    #axes[1].imshow(nuclei_markers)

    # Identify peaks =========================================================
    # identify transcription particles, diameter of 3 works well
    # this params seem to work decently to identify single transcripts
    # Imaging params: LeicaImagingFacility, 100%int 300msExp 0.2uZstack
    _parts = tp.locate(fish, 3, minmass=45)
    _parts['imname'] = imname
    # from the ecdf of signal int, this separates perfectly single transcripts
    # from center of TSS blobs
    tssblobs = _parts[_parts.signal>120]

    # Assign transcripts to cells ====================================================

    # Get cell label
    _parts['cell_label'] = _parts.apply(lambda coords:\
            cell_markers_enlarged[int(coords.y), int(coords.x)], axis=1)
    # Get nuclear label. This indicates whether inside nucleus or not
    _parts['nuc_label'] = _parts.apply(lambda coords:\
            nuclei_markers[int(coords.y), int(coords.x)], axis=1)
    # remove peaks that are not in cells or inside nuclei
    _parts_bk = _parts.copy()
    _parts = _parts[(_parts.cell_label>0)&(_parts.nuc_label==0)]

    ax = next(axes)
    tp.annotate(_parts, fish_viz, ax=ax, imshow_style={'cmap':'viridis'})

    #fig, axes = plt.subplots(2, sharex=True, sharey=True)
    #tp.annotate(_parts_bk, fish_viz, ax=axes[0], imshow_style={'cmap':'viridis'})
    #tp.annotate(_parts, fish_viz, ax=axes[1], imshow_style={'cmap':'viridis'})

    peaks = pd.concat((peaks, _parts))
