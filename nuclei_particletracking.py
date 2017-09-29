from matplotlib import pyplot as plt
%matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import skimage
from skimage import io
import os
from im_utils import *
import trackpy as tp
from tqdm import tqdm
from collections import defaultdict

# load z-projected movies
movs = {}
ddir = '../data/GFPEnvymScarlet/raw_projected/'
for fname in tqdm(os.listdir(ddir)):
    if 'tiff' in fname:
        movs[fname.split('.')[0]] = io.imread(ddir + fname)
    else: next

# area and intensity limits
maxint_lim, minor_ax_lim, major_ax_lim, area_lim = (0.2,1), (25, 500), (40, 500), (500, 5000)

# dictionary for time-projected mask, nuclei markers and properties
nuclei_proj = {}
# dataframe for transcription peaks and nuclei properties
nuclei_peaks = pd.DataFrame()

for mname in tqdm(movs):
    print('analyzing {}'.format(mname))
    movie = movs[mname]
    print('normalizing...')
    movie_norm = normalize(movie) #np.stack([normalize(frame) for frame in movie])
    print('projecting through time axis...')
    movie_proj = z_project(movie_norm, 'max')
    print('creating binary mask...')
    # mask projected image using adaptive threshold to find nuclei of min_size
    m_mask = mask_image(movie_proj, min_size=200, block_size=101, selem=skimage.morphology.disk(15))
    print('filtering...')
    # get nuclei markers, bound by area and intensity
    markers_proj, _nuclei_proj = label_sizesel(movie_proj, m_mask,
                        maxint_lim, minor_ax_lim, major_ax_lim, area_lim)
    print('identifying peaks...')
    # identify transcription particles, diameter of 3 works well
    _parts = tp.batch(movie, 3)
    # enlarge nuclei markers to keep particles close to nuclear edge
    markers_proj_enlarged = skimage.morphology.dilation(markers_proj,
            selem=skimage.morphology.disk(5))
    # Get nuclear label
    _parts['label'] = _parts.apply(lambda coords:\
            markers_proj_enlarged[int(coords.y), int(coords.x)], axis=1)
    # remove peaks that are not in identified nuclei
    _parts = _parts[_parts.label>0]
    # track them
    print('tracking particles...')
    _parts = tp.link_df(_parts, 5, memory=1)
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
    _nuclei_peaks['movie'] = mname
    # transform particle whole movie coordinates to nuclei bounding box coordinates
    _nuclei_peaks['bbx'] = _nuclei_peaks['x'] - _nuclei_peaks.bbox.apply(lambda x: x[1])
    _nuclei_peaks['bby'] = _nuclei_peaks['y'] - _nuclei_peaks.bbox.apply(lambda x: x[0])
    # get bounding box image
    nuclei_peaks['bb_image'] = [movie[x.frame, x.bbox[0]:x.bbox[2],
                x.bbox[1]:x.bbox[3]] for (_, x) in nuclei_peaks.iterrows()]

    nuclei_peaks = pd.concat([nuclei_peaks, _nuclei_peaks])
    # save markers and time-projected properties
    nuclei_proj[mname] = (_nuclei_proj, markers_proj)
    _nuclei['movie'] = mname

# save dataframe as pickle to preserve numpy arrays
nuclei_peaks.to_pickle('nuclei_peaks.p')
