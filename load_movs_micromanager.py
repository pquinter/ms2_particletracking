"""
batch maximum intensity z-projection of movies
For movies taken with 3 different stage positions
run as:
    python load_movs.py dir_to_load
where dir_to_load contains multiple directories, each with groups of 3 movies
if multistage movie not found, try loading single movie
"""
import numpy as np
from im_utils import z_project
from skimage import io
from skimage.external.tifffile import TiffFile
from sys import argv
from os import mkdir, listdir
import glob
from tqdm import tqdm
import warnings

rootdir = argv[1] + '/'
#datadir = '/Volumes/QuinDrive/Research/Data/YeastMicroscopy/' + rootdir
datadir = '/Users/porfirio/Documents/research/sternberg_lab/yeastEP/ms2pp7/data/PP7/'+rootdir
patterns = [datadir + d + '/Pos0/*tif' for d in listdir(datadir) if 'DS' not in d]

savedir = '../data/' + rootdir[:-1] + '_projected/'
for ldir in tqdm(patterns):
    # get name to save
    sdir = ldir.split('/')[-3] + '/'
    # get file names, number of stacks, frames and timepoints
    fnames = glob.glob(ldir)
    no_stacks = len(set(s.split('_')[-1] for s in fnames))
    no_frames = len(fnames)
    if no_frames<10: continue
    try:
        no_timepoints = int(no_frames/no_stacks)
    except ZeroDivisionError:
        continue
    # load every movie
    coll = io.ImageCollection(ldir)
    # group by z-series and project
    n = 0
    proj_frames = np.empty((no_timepoints,) + coll[0].shape, dtype=coll[0].dtype)
    for i in range(no_timepoints):
        proj_frames[i] = z_project(zseries for zseries in coll[n:n+no_stacks])
        n += no_stacks
    # create save directory if it doesn't exist yet
    try: mkdir(savedir+sdir)
    except FileExistsError: pass
    fname = ldir.split('/')[-2]
    saveto = savedir + sdir + fname + '_zproj.tif'
    print('processing {} and saving to {}'.format(ldir, saveto))
    io.imsave(saveto, proj_frames)
