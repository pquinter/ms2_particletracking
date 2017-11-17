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

# dictionary for time-projected mask, nuclei markers and properties
nuclei_proj = {}
# dataframe for transcription peaks and nuclei properties
peaks = pd.DataFrame()

def segment_cellnuc(im_cells, im_nuclei):
    """
    Segment cells/nuclei
    """
    # mask projected image using adaptive threshold
    mask_cells = mask_image(im_cells, min_size=100, block_size=101,
                            selem=skimage.morphology.disk(10))
    mask_nuclei = mask_image(im_nuclei, min_size=100, block_size=101,
                            selem=skimage.morphology.disk(10))
    # create joint cell+nuclei mask
    joint_mask = mask_nuclei.astype('int') + mask_cells
    # update masks to keep only cells with nuclei and viceversa
    mask_cells = skimage.morphology.reconstruction(mask_nuclei, joint_mask)
    mask_nuclei = mask_nuclei * mask_cells

    # normalize to reuse bounds that worked before
    cells_norm = normalize_im(im_cells)
    # get cell markers and properties, bound by area and intensity
    cells_markers, cells_props = label_sizesel(cells_norm, mask_cells,
                    maxint_lim, minor_ax_lim, major_ax_lim, area_lim)
    # use the same labels for nuclei and measure region props, too
    nuclei_markers = cells_markers * mask_nuclei
    nuclei_props = skimage.measure.regionprops(nuclei_markers)

    # enlarge markers and merge region properties by label
    markers_enlarged, props = [], []
    for _markers, _props in ((cells_markers, cells_props),
                            (nuclei_markers, nuclei_props)):
        props.append(pd.concat([regionprops2df(p) for p in _props]))
        markers_enlarged.append(skimage.morphology.dilation(_markers,
                selem=skimage.morphology.disk(5)))
    props = pd.merge(*props, on='label', suffixes=('_cell','_nuc'))

    return markers_enlarged, cells_markers, nuclei_markers, mask_cells, mask_nuclei, props

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

    # segment cells ===========================================================
    print('segmenting cells...')
    cell_markers_enlarged, _cells_props, mask_cells = segment_cell(autof)

    # segment nuclei ==========================================================
    print('segmenting nuclei...')
    nuclei_markers_enlarged, _nuclei_props, mask_nuclei = segment_cell(dapi)

    (cells_markers_enlarged, nuclei_markers_enlarged), cell_markers,
    nuclei_markers, mask_cells, mask_nuclei, props = segment_cellnuc(autof, dapi)


    fig, axes = plt.subplots(2, sharex=True, sharey=True)
    axes[0].imshow(cells_markers_enlarged)
    axes[1].imshow(nuclei_markers_enlarged)

    # Identify peaks =========================================================
    # identify transcription particles, diameter of 3 works well
    # this params seem to work decently to identify single transcripts
    # Imaging params: LeicaImagingFacility, 100%int 300msExp 0.2uZstack
    _parts = tp.locate(fish, 3, minmass=45)
    _parts['imname'] = imname
    # from the ecdf of signal int, this separates perfectly single transcripts
    # from center of TSS blobs
    tssblobs = _parts[_parts.signal>120]

    ax = next(axes)
    tp.annotate(_parts, fish_viz, ax=ax, imshow_style={'cmap':'viridis'})
    peaks = pd.concat((peaks, _parts))

    # Assign transcripts to cells ====================================================
    # Remove cells without nuclei and viceversa
    # sum masks, cells with nuclei must now contain values of 2
    joint_mask = mask_nuclei.astype('int')+mask_cells
    # nuclei must be (almost) entirely contained within cells
    # get bounding box image from joint mask
    _nuclei_props['joint_mask'] = [joint_mask[x.bbox[0]:x.bbox[2],
                x.bbox[1]:x.bbox[3]] for (_, x) in _nuclei_props.iterrows()]
    # 





#    # Get nuclear label
#    _parts['label'] = _parts.apply(lambda coords:\
#            markers_proj_enlarged[int(coords.y), int(coords.x)], axis=1)
#    # remove peaks that are not in identified nuclei
#    _parts = _parts[_parts.label>0]
#    # Dataframe for movie nuclei properties
#    _nuclei = pd.DataFrame()
#    print('measuring nuclei properties...')
#    for frame_no, frame in enumerate(movie):
#        # commented lines can be used to create a frame specific mask, which is
#        # a more correct approach, but it doesn't really change much and adds
#        # computation time (~3s per loop)
#        ## create frame mask
#        #mask_f = mask_image(frame, min_size=200, block_size=101, selem=skimage.morphology.disk(10))
#        ## use time-projected, filtered markers to mark frame mask
#        #markers_f = mask_f * markers_proj
#        #nuclei_f = skimage.measure.regionprops(markers_f, frame)
#        # get nuclei for each frame based on time-proj markers
#        nuclei_f = skimage.measure.regionprops(markers_proj, frame)
#        # convert to dataframe and save
#        nuclei_fdf = regionprops2df(nuclei_f)
#        nuclei_fdf['frame'] = frame_no
#        _nuclei = pd.concat([_nuclei, nuclei_fdf])
#    print('{0} total nuclei for {1}'.format(len(nuclei_f), mname))
#
#    # Merge, label and save.
#    # Particles can be matched to nuclei based on frame number and nuclear label
#    _nuclei_peaks = pd.merge(_nuclei, _parts, on=['frame', 'label'])
#    _nuclei_peaks['movie'] = mname
#    # transform particle whole movie coordinates to nuclei bounding box coordinates
#    _nuclei_peaks['bbx'] = _nuclei_peaks['x'] - _nuclei_peaks.bbox.apply(lambda x: x[1])
#    _nuclei_peaks['bby'] = _nuclei_peaks['y'] - _nuclei_peaks.bbox.apply(lambda x: x[0])
#    # get bounding box image
#    _nuclei_peaks['bb_image'] = [movie[x.frame, x.bbox[0]:x.bbox[2],
#                x.bbox[1]:x.bbox[3]] for (_, x) in _nuclei_peaks.iterrows()]
#
#    nuclei_peaks = pd.concat([nuclei_peaks, _nuclei_peaks])
#    # save markers and time-projected properties
#    nuclei_proj[mname] = (_nuclei_proj, markers_proj)
#    _nuclei['movie'] = mname
#
## save dataframe as pickle to preserve numpy arrays
#nuclei_peaks.to_pickle('nuclei_peaks.p')
