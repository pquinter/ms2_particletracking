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
#movs_smooth = {}

ddir = '../data/PP7/PP7_HJ_projected/'
for fname in tqdm(os.listdir(ddir)):
    if 'tif' in fname:
        _mov = io.imread(ddir + fname)
        _mname = fname.split('.')[0]
        movs[_mname] = _mov
    else: next

# area and intensity limits
maxint_lim, minor_ax_lim, major_ax_lim, area_lim = (0.1,1), (15, 500), (20, 500), (50, 5000)

# dataframe for transcription peaks and nuclei properties
peaks_complete = pd.DataFrame()

for mname in tqdm(movs):
    print('analyzing {}'.format(mname))
    movie = movs[mname]

    # segment cells ===========================================================
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
    markers_proj, reg_props = label_sizesel(movie_proj, m_mask,
            maxint_lim, minor_ax_lim, major_ax_lim, area_lim, watershed=True)
    # convert nuclei region properties to dataframe
    reg_props = regionprops2df(reg_props, props=('label','centroid'))
    # expand centroid coordinates into x,y cols
    reg_props[['y_cell','x_cell']] = reg_props.centroid.apply(pd.Series)
    # enlarge nuclei markers to keep particles close to nuclear edge
    markers_proj_enlarged = skimage.morphology.dilation(markers_proj,
            selem=skimage.morphology.disk(5))

    # identify and track parts ================================================
    print('identifying peaks...')
    # identify transcription particles, diameter of 3 works well
    _parts = tp.batch(movie, 3, minmass=1000, characterize=False)
    # Get nuclear label
    _parts['label'] = _parts.apply(lambda coords:\
            markers_proj_enlarged[int(coords.y), int(coords.x)], axis=1)
    # track them, search range of 5 pixels, remember particle for 5 frames
    print('tracking particles...')
    _parts = tp.link_df(_parts, 5, memory=5)

    # Tidy up ==========================================================
    # Relabel particles that were in cells and left (e.g. close to edge)
    for plabel, group in _parts.groupby('particle'):
        c_label = group.sort_values('label', ascending=False).label.values[0]
        _parts.loc[(_parts.particle==plabel)&(_parts.label<1), 'label']=c_label
    # remove peaks that never were in cells
    _parts = _parts[_parts.label>0]
    # if more than one spot detected in the same cell and frame, keep brightest
    _parts = _parts.sort_values('mass', ascending=False).drop_duplicates(['frame', 'label'])
    # merge particles of the same cell
    for label, group in _parts.groupby('label'):
        # assign the most common particle label to all
        plabels = group.particle.values
        plabels_counts = np.unique(plabels, return_counts=True)
        group_plabel = plabels[np.argmax(plabels_counts)]
        # remove particles that appear in other cells
        if group_plabel in _parts[~(_parts.label==label)].particle.values:
            _parts = _parts.loc[~(_parts.particle.isin(plabels))]
            continue
        else:
            _parts.loc[_parts.label==label, 'particle'] = group_plabel
    # keep only trajectories of more than 10 frames
    _parts = tp.filter_stubs(_parts, 10)

    # fill in int values of missing frames ==================================
    _parts = pd.merge(reg_props[['label','x_cell','y_cell']],
                                    _parts, how='outer', on='label')
    _peaks_complete = pd.DataFrame()
    for label, group in _parts.groupby('label'):
        coords_df = pd.DataFrame()
        coords_df['frame'] = np.arange(0, len(movie))
        coords_df['imname'] = mname
        coords_df = pd.merge(group, coords_df, on='frame', how='right')
        # if no particles in cell, need to actively preserve x,y coords
        if coords_df.x.isnull().values[0]:
            coords_df['x_cell'] = group['x_cell'].values[0]
            coords_df['y_cell'] = group['y_cell'].values[0]
            coords_df['label'] = label
        # save 'mass' series to assign nan later to nonparticles later
        mass_series = coords_df['mass']
        # fill first frame if missing to be able to do forward fill, see below
        if coords_df.loc[coords_df.frame==0].x.isnull().values[0]:
            coords_df.loc[coords_df.frame==0] = coords_df.fillna(method='bfill')[coords_df.frame==0]
        # fill rest of missing frames using coords of last observed part
        coords_df = coords_df.fillna(method='ffill')
        try:
            part_ims = get_batch_bbox(coords_df, {mname:movie_norm}, movie=True)
        except ValueError: # no particles in cell, still get intensity
            part_ims = get_batch_bbox(coords_df, {mname:movie_norm}, 
                    movie=True, coords_col=['x_cell','y_cell'])
        int_vals = [np.max(im) for im in part_ims]
        # assign intensity and mass of nan to non particles
        coords_df['intensity'] =  int_vals
        coords_df['mass'] =  mass_series
        _peaks_complete = pd.concat((_peaks_complete, coords_df.sort_values('frame')), ignore_index=True)
    # Reassign mname. Necessary because cells without parts dont' have it
    _peaks_complete['imname'] = mname
    peaks_complete = pd.concat((peaks_complete, _peaks_complete), ignore_index=True)
peaks_complete['pid'] = peaks_complete.apply(lambda x: str(x.particle)+'_'+x.imname, axis=1)
peaks_complete.to_csv('../output/pp7/peaks_complete.csv', index=False)
