import pandas as pd
import numpy as np
from skimage import io
import im_utils

from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

import trackpy as tp
import re
from utils import image
import os
import glob
import pickle

##############################################################################
# particle detection
##############################################################################

def locate_par(frame_no, frame, diameter, **kwargs):
    """ Parallelizable particle locate function from trackpy """
    part_df = tp.locate(frame, diameter, **kwargs)
    part_df['frame'] = frame_no
    return part_df

def locate_batch_par(mov, diameter=3, n_jobs=multiprocessing.cpu_count(), **kwargs):
    """ Parallelized batch particle locate with trackpy """
    parts = Parallel(n_jobs=n_jobs)(delayed(locate_par)(i, frame, diameter, **kwargs)
            for i, frame in tqdm(enumerate(mov), desc='Detecting peaks'))
    parts = pd.concat(parts).reset_index(drop=True)
    return parts

def group_getroi(df, mask, n_jobs=multiprocessing.cpu_count()):
    """ Parallelized get ROI from mask for grouped dataframe """

    def getroi(group, mask):
        """ Get roi value from single or movie of masks """
        # single mask
        if mask.ndim==2:
            return group.apply(lambda coords: mask[int(coords.y), int(coords.x)], axis=1)
        # or movie of masks
        elif mask.ndim==3:
            return group.apply(lambda coords: mask[int(coords.frame)][int(coords.y), int(coords.x)], axis=1)
        else: raise ValueError('Mask of wrong shape: {}'.format(mask.shape))

    app_list = Parallel(n_jobs=n_jobs)(delayed(getroi)(group, mask)
        for name, group in tqdm(df.groupby('frame'), desc='masking dataframe'))

    return pd.concat(app_list).values

def locate_batch(mov_dir, movie=True):

    ################################################################################
    # Load data
    ################################################################################

    mov = io.imread(mov_dir)
    mov_name = re.search(r'.+/(.+)(?:\.tif)$', mov_dir).group(1)
    laser_power = int(re.search(r'(\d{3})u.*?480', mov_name).group(1))

    # get ROIs
    if movie:
        markers_dir = '../output/pipeline/segmentation/{}.tif'.format(mov_name)
        markers = io.imread(markers_dir)
    else:
        markers_dir = '../output/pipeline_snapshots/segmentation/{}.tif'.format(mov_name)
        markers = io.imread(markers_dir)
        # make a one movie frame with image for everything else to work
        mov = np.stack([mov])
    rois = image.markers2rois(markers)

    ################################################################################
    # Detect transcription sites and assign cell label
    ################################################################################

    # identify transcription particles in parallel
    locate_kwargs = {'minmass':25, 'characterize':True, 'noise_size':1, 'smoothing_size':5}
    parts = locate_batch_par(mov, diameter=3, **locate_kwargs)
    # make markers and get median fluor with manually selected ROIs for each frame
    markers, nuc_fluor = image.refine_markers(rois, mov)
    # assign roi number to each spot; parallelized is much faster
    parts['roi'] = group_getroi(parts, markers)
    # and metadata
    parts['laser_power'] = laser_power
    parts['mov_name'] = mov_name
    # add median nuclear fluorescence
    parts = pd.merge(parts, nuc_fluor, on=['roi','frame'])
    # raw mass to nuclear fluorescence ratio
    parts['mass_norm'] = parts.raw_mass.values / parts.nuc_fluor.values
    parts['strain'] = parts.mov_name.apply(lambda x: re.search(r'_(((yQC)|(TL)).+?)_', x).group(1))

    return parts

def cleanup_track(parts, traj_len_thresh=1, part_per_frame=1, remove_hotpixels=10):
    # make sure df belongs to single movie
    assert len(parts.mov_name.unique())==1
    # remove peaks outside cells
    parts_filt = parts[parts.roi>0]
    # remove hot pixels
    if remove_hotpixels:
        parts_filt = parts_filt[parts_filt.mass<parts_filt.mass.mean()*remove_hotpixels]
    # keep two brightest spots per ROI per frame; 'tail' method allows to keep top N
    parts_filt = parts_filt.sort_values('mass').groupby(['roi','frame']).tail(part_per_frame+1).reset_index(drop=True)
    # link particles with search range of 5 px and memory of 1 frame
    parts_filt = tp.link_df(parts_filt, 5, memory=1)
    # compute trajectory length for each particle
    traj_len = parts_filt.groupby('particle').count().reset_index()[['particle','x']]
    traj_len.columns = ['particle','traj_len']
    parts_filt = pd.merge(parts_filt, traj_len, on='particle')
    # remove trajectories too long to be true
    parts_filt = parts_filt[parts_filt.traj_len<parts_filt.traj_len.mean()*10]
    # filter spurious spots by trajectory length; faster than tp.filter_stubs
    parts_filt = parts_filt[parts_filt.traj_len>traj_len_thresh]
    # keep only one spot per ROI per frame: brightest or with longest traj
    parts_filt = parts_filt.sort_values(['mass','traj_len']).groupby(['roi','frame']).tail(part_per_frame).reset_index(drop=True)
    # assign unique particle id
    parts_filt['pid'] = parts_filt['mov_name']+'_'+parts_filt['particle'].apply(str)+'_'+parts_filt['frame'].apply(str)
    return parts_filt

def get_patches(mov_path, patch_dir, parts, movie=True, n_jobs=multiprocessing.cpu_count()):
    """ Load movie, smooth and retrieve spot patches from raw and smooth """
    # default params
    radius, noise_size, smoothing_size, threshold, im_size = 1.5, 1, 5, 1, 15
    # check if already analyzed this movie
    print('Analyzing {0}'.format(mov_path))
    mov_name = mov_path.split('/')[-1][:-4] # load movie to memory
    fname_ims = '{}/{}_15x15spots.p'.format(patch_dir, mov_name)
    if os.path.isfile(fname_ims):
        print('already analyzed this one, moving on...')
        return None
    raw_mov = io.imread(mov_path)
    if not movie: raw_mov = np.stack([raw_mov])
    bp_mov = np.stack(Parallel(n_jobs=n_jobs)(delayed(tp.bandpass)(f, noise_size, smoothing_size, threshold)
            for f in tqdm(raw_mov, desc='applying bandpass filter')))
    # get particles of current movie
    print('retrieving spot images...')
    _parts = parts[parts.mov_name==mov_name]
    raw_ims = im_utils.get_batch_bbox(_parts, raw_mov, size=im_size, movie=True)
    # delete movies from memory! (assign None) otherwise computer breaks...
    raw_mov = None
    bp_ims = im_utils.get_batch_bbox(_parts, bp_mov, size=im_size, movie=True)
    bp_mov = None
    print('pickling...')
    with open(fname_ims, 'wb') as f:
        pickle.dump(_parts.pid.values, f)
        pickle.dump(raw_ims, f)
        pickle.dump(bp_ims, f)
    raw_ims, bp_ims = None, None
    print('done')

##############################################################################
# particle annotation and classification
##############################################################################

def load_patches(spots_dir, shape='auto', filter_trunc=True):
    """
    Load spot image patches
    spots_dir: str
        directory containing all pickled spot images by movie
    shape: tuple, auto
        tuple with shape of image patch to check for
        auto: infer most common shape from images
    filter_trunc: bool
        whether to filter out images truncated images
    """
    if '.p' not in spots_dir:
        spots_path = glob.glob(spots_dir+'/*.p')
    else: spots_path = [spots_dir]
    pids_all, rawims_all, bpims_all = [],[],[]
    for _path in spots_path:
        mov_name = _path.split('/')[-1][:-13]
        try:
            with open(_path, 'rb') as f:
                pids_all.extend(pickle.load(f))
                rawims_all.extend(pickle.load(f))
                bpims_all.extend(pickle.load(f))
        except EOFError: # try to load without pid
            pids_all, rawims_all, bpims_all = [],[],[]
            with open(_path, 'rb') as f:
                rawims_all.extend(pickle.load(f))
                bpims_all.extend(pickle.load(f))
    if filter_trunc:
        if shape=='auto':
            # get most common (median) image shape
            shape_all = np.array([im.shape[-2:] for im in rawims_all])
            shape = tuple([np.median(shape_all[:,i]).astype(int) for i in range(shape_all.ndim)])
        for arr in (rawims_all, bpims_all, pids_all):
            # filter out all truncated images
            is_full = [im.shape==shape for im in arr]
            arr = np.array(arr)[is_full]
    # Turn them into concatenated arrays
    try:
        rawims_all, bpims_all = [np.stack(a) for a in (rawims_all, bpims_all)]
    except TypeError: pass
    return pids_all, rawims_all, bpims_all

def get_manual_labels(sampled_frames, mov_name, mov_labeled, parts, brush_val=0, verbose=True):
    """
    Get which spots were labeled as True in sample movies edited in Fiji
    True spots have a paintbrush on top

    Arguments
    ---------

    sampled_frames: DataFrame
        contains movie name and original frame number; index is sample movie frame
    mov_name: str
    mov_labeled: array
    parts: DataFrame
        whole particle dataframe to add labels to
    brush_val: int
        Fiji paintbrush value used to label spots (black or 0 is easy to detect)
    verbose: bool
        whether to print how many spots were labeled as True

    Returns
    ---------
    _parts: DataFrame
        copy of slice of parts DataFrame with manual labels
    """
    # get frame numbers from sample movie and corresponding original movie
    sample_fno = sampled_frames.index.values
    orig_fno = sampled_frames.frame.values
    # get relevant particle data
    _parts = parts[(parts.mov_name==mov_name)&(parts.frame.isin(orig_fno))].copy()
    # map sampled frames to original movie frame number
    rep_values = dict(zip(orig_fno, sample_fno))
    _parts['sframe'] = _parts.frame.map(rep_values)
    # get intensity value from labeled movie
    _parts['int_value'] = _parts.apply(lambda coords:\
            mov_labeled[int(coords.sframe), int(coords.y), int(coords.x)], axis=1)
    # get label (spots were labeled in Fiji, by drawing a paintbrush)
    _parts['manual_label'] = _parts.int_value.values == brush_val
    if verbose:
        man_count = _parts.manual_label.sum()
        print("For Movie {0} {1} were labeled".format(mov_name, man_count))
    del _parts['int_value']
    return _parts

def get_cix(start, stop, step):
    """ Generator of two by two indices to iterate in chunks of size `step` """
    return tqdm(zip(np.arange(start, stop, step), np.arange(start+step, stop+step, step)))

def predict_prob(df, feat, clf_scaler_path, chunk_size=1e4,
        n_jobs=multiprocessing.cpu_count()):
    """
    Classify spots from dataframe

    Arguments
    ---------
    df: dataframe
        Must contain columns `feat`
    feat: list
        features to be classified (e.g. mass and correlation_with_idealspot)
    clf_scaler_path: str
        path to pickled classifier and scaler objects
    chunk_size: int
        predict probability of slices this size
    n_jobs: int
        number of jobs for parallel processing

    Returns
    -------
    prob_pred: array
        predicted probabilities of being True
    """

    with open(clf_scaler_path, 'rb') as f:
        clf = pickle.load(f)
        scaler = pickle.load(f)
    # scale features like training data
    feat_scaled = scaler.transform(df[feat].values)
    # predict probability in chunks, otherwise fails
    # Defualt predicted label is just thresholded by >=50% prob, don't do that
    chunk_size = int(chunk_size)
    prob_pred = Parallel(n_jobs=n_jobs)(delayed(clf.predict_proba)(feat_scaled[t_:_t])
            for t_, _t in tqdm(get_cix(0, len(feat_scaled), chunk_size)))
    prob_pred = np.concatenate(prob_pred)
    return prob_pred[:,1] # prob of being True, first col is 1-prob
