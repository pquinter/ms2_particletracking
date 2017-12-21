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
movs = {}
movs_smooth = {}

ddir = '../data/PP7/PP7_HJ_projected/unloaded/'
for fname in tqdm(os.listdir(ddir)):
    if 'tif' in fname:
        _mov = io.imread(ddir + fname)
        _mname = fname.split('.')[0]
        movs[_mname] = _mov
        # Convert image to float
        _mov_norm = normalize_im(_mov)
        # substract background with strong gaussian blur
        _mov_bg = skimage.filters.gaussian(_mov_norm, sigma=50)
        _mov_norm -= _mov_bg
        # smooth with gaussian
        _mov_smooth = skimage.filters.gaussian(_mov_norm, sigma=1)
        movs_smooth[_mname] = _mov_smooth
    else: next
# remove crappy movie
#del movs['12032017_TL47_time_4']
#del movs_smooth['12032017_TL47_time_4']

# area and intensity limits
maxint_lim, minor_ax_lim, major_ax_lim, area_lim = (0.1,1), (15, 500), (20, 500), (500, 5000)

# dictionary for time-projected mask, nuclei markers and properties
nuclei_proj = {}
# dataframe for transcription peaks and nuclei properties
nuclei_peaks = pd.DataFrame()

for mname in tqdm(movs):
    #test on a single movie
    if mname!='20171219_67f6_Late': continue
    print('analyzing {}'.format(mname))
    movie = movs[mname]
    print('normalizing...')
    movie_norm = normalize(movie) #np.stack([normalize(frame) for frame in movie])
    print('projecting through time axis...')
    movie_proj = z_project(movie_norm, 'max')
    print('creating binary mask...')
    # mask projected image using adaptive threshold to find nuclei of min_size
    # reduce diameter of selem to detect more
    m_mask = mask_image(movie_proj, min_size=100, block_size=101, 
            selem=skimage.morphology.disk(5), clear_border=False)
    print('filtering...')
    # get nuclei markers with watershed transform, bound by area and intensity
    markers_proj, _nuclei_proj = label_sizesel(movie_proj, m_mask,
            maxint_lim, minor_ax_lim, major_ax_lim, area_lim, watershed=True)
    print('identifying peaks...')
    # identify transcription particles, diameter of 3 works well
    _parts = tp.batch(movie, 3, minmass=1000)
    # enlarge nuclei markers to keep particles close to nuclear edge
    markers_proj_enlarged = skimage.morphology.dilation(markers_proj,
            selem=skimage.morphology.disk(5))
    # Get nuclear label
    _parts['label'] = _parts.apply(lambda coords:\
            markers_proj_enlarged[int(coords.y), int(coords.x)], axis=1)
    # track them, search range of 10, remember particle for 5 frames
    print('tracking particles...')
    _parts = tp.link_df(_parts, 10, memory=5)
    # merge particles of the same cell
    _parts_ = pd.DataFrame()
    for label, group in _parts.groupby('label'):
        # if more than one spot detected in frame, keep brightest
        group = group.sort_values('signal', ascending=False).drop_duplicates('frame')
        # assign the most common particle label to all
        plabels = group.particle.values
        plabels_counts = np.unique(plabels, return_counts=True)
        group['particle'] = plabels[np.argmax(plabels_counts)]
        _parts_ = pd.concat((_parts_, group))
    _parts, _parts_ = _parts_, pd.DataFrame()
    # relabel particles that were in cells and were undetected in some frames
    for plabel, group in _parts.groupby('particle'):
        cell_label = group.sort_values('label', ascending=False).label.values[0]
        group.loc[group.label<1].label = cell_label
        _parts_ = pd.concat((_parts_, group))
    _parts = _parts_
    # remove peaks that never were in cells
    _parts = _parts[_parts.label>0]
    # filter, keep only trajectories of more than 2
    _parts = tp.filter_stubs(_parts, 5)
    # Dataframe for movie nuclei properties
    _nuclei = pd.DataFrame()
    print('measuring nuclei properties...')
    for frame_no, frame in enumerate(movie):
        # commented lines can be used to create a frame specific mask, which is
        # a more correct approach, but it doesn't really change much and adds
        # computation time (~3s per loop)
        ## create frame mask
        #mask_f = mask_image(frame, min_size=200, block_size=101, selem=skimage.morphology.disk(10))
        ## use time-projected, filtered markers to mark frame mask
        #markers_f = mask_f * markers_proj
        #nuclei_f = skimage.measure.regionprops(markers_f, frame)
        # get nuclei for each frame based on time-proj markers
        nuclei_f = skimage.measure.regionprops(markers_proj, frame)
        # convert to dataframe and save
        nuclei_fdf = regionprops2df(nuclei_f)
        nuclei_fdf['frame'] = frame_no
        _nuclei = pd.concat([_nuclei, nuclei_fdf])
    print('{0} total nuclei for {1}'.format(len(nuclei_f), mname))

    # Merge, label and save.
    # Particles can be matched to nuclei based on frame number and nuclear label
    _nuclei_peaks = pd.merge(_nuclei, _parts, on=['frame', 'label'])
    _nuclei_peaks['imname'] = mname
    # add movie specific particle id
    _nuclei_peaks['pid'] = _nuclei_peaks.apply(lambda x: str(x.particle)+x.imname, axis=1)
    # transform particle whole movie coordinates to nuclei bounding box coordinates
    _nuclei_peaks['bbx'] = _nuclei_peaks['x'] - _nuclei_peaks.bbox.apply(lambda x: x[1])
    _nuclei_peaks['bby'] = _nuclei_peaks['y'] - _nuclei_peaks.bbox.apply(lambda x: x[0])
    # get bounding box image
    _nuclei_peaks['bb_image'] = [movie[x.frame, x.bbox[0]:x.bbox[2],
                x.bbox[1]:x.bbox[3]] for (_, x) in _nuclei_peaks.iterrows()]

    nuclei_peaks = pd.concat([nuclei_peaks, _nuclei_peaks])
    # save markers and time-projected properties
    nuclei_proj[mname] = (_nuclei_proj, markers_proj)

# save dataframe as pickle to preserve numpy arrays
nuclei_peaks.to_pickle('../output/pp7/nuclei_peaks.p')
