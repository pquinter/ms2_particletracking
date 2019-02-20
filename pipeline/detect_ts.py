from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import trackpy as tp
import pickle
import utils
from skimage import io
import skimage
import re
from im_utils import concat_movies
import warnings
import glob
from joblib import Parallel, delayed
from tqdm import tqdm

def locate_track_parts(mov_dir):

    ################################################################################
    # Load data
    ################################################################################

    mov = io.imread(mov_dir)
    mov_name = re.search(r'.+/(.+)(?:\.tif)$', mov_dir).group(1)
    laser_power = int(re.search(r'(\d{3})u.*?480', mov_name).group(1))
    roi_dir = '{0}/manual_cell_rois/{1}.p'.format(root_dir, mov_name)
    try:
        with open(roi_dir, 'rb') as f:
            roi_mov = pickle.load(f)
            rois = pickle.load(f)
    except FileNotFoundError:
        warnings.warn('Could not find ROIs for {}'.format(mov_name))

    ################################################################################
    # Detect, filter and track transcription sites
    ################################################################################

    # identify transcription particles in parallel
    locate_kwargs = {'minmass':25, 'characterize':True, 'noise_size':1, 'smoothing_size':5}
    _parts = utils.locate_batch_par(mov, diameter=3, **locate_kwargs)
    # make mask and get median fluor with manually selected ROIs for each frame
    mask_mov, nuc_fluor = utils.mask_rois_mov(rois, mov)
    # assign roi number to each spot; parallelized is much faster
    _parts['roi'] = utils.group_getroi(_parts, mask_mov)
    # remove spots outside ROIs
    _parts_filt = _parts[_parts.roi>0].copy()
    # keep brightest spot per ROI per frame; 'tail' method allows to keep top N
    _parts_filt = _parts_filt.sort_values('mass').groupby(['roi','frame']).tail(1).reset_index(drop=True)
    # link particles with search range of 5 px and memory of 1 frame
    _parts_filt = tp.link_df(_parts_filt, 5, memory=1)
    # compute trajectory length for each particle
    traj_len = _parts_filt.groupby('particle').count().reset_index()[['particle','x']]
    traj_len.columns = ['particle','traj_len']
    _parts_filt = pd.merge(_parts_filt, traj_len, on='particle')
    # keep only spots that are at least 2 frames long; faster than tp.filter_stubs
    _parts_filt = _parts_filt[_parts_filt.traj_len>1]
    # add median nuclear fluorescence
    _parts_filt = pd.merge(_parts_filt, nuc_fluor, on=['roi','frame'])
    # and metadata
    _parts_filt['laser_power'] = laser_power
    _parts_filt['mov_name'] = mov_name
    # clear memory
    _parts = None
    return _parts_filt

root_dir = '../data/2019_galinducedt0/'
movdir_list = glob.glob(root_dir+'*tif')
parts = Parallel(n_jobs=12)(delayed(locate_track_parts)(mov_dir)
            for mov_dir in tqdm(movdir_list))
# remove hot pixels
parts = parts[parts.mass<parts.mass.mean()*10]
# and trajectories too long to be true
parts = parts[parts.traj_len<parts.traj_len.mean()*10]
parts = pd.concat(parts, ignore_index=True)
parts['strain'] = parts.mov_name.apply(lambda x: re.search(r'_(((yQC)|(TL)).+?)_', x).group(1))
parts['mass_norm'] = parts.raw_mass.values / parts.nuc_fluor.values
parts['pid'] = parts['mov_name']+'_'+parts['particle'].apply(str)+'_'+parts['frame'].apply(str)
parts.to_csv('../output/02182019_parts.csv', index=False)


