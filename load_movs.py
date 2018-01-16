"""
batch maximum intensity z-projection of movies
For movies taken with 3 different stage positions
run as:
    python load_movs.py dir_to_load
where dir_to_load contains multiple directories, each with groups of 3 movies
if multistage movie not found, try loading single movie
"""
import numpy as np
from im_utils import z_project, load_zproject_STKcollection
from skimage import io
from skimage.external.tifffile import TiffFile
from sys import argv
from os import mkdir, listdir
import glob
from tqdm import tqdm
import warnings

rootdir = argv[1] + '/'
datadir = '/Volumes/QuinDrive/Research/Data/YeastMicroscopy/' + rootdir
dirs_toload = [datadir + d + '/' for d in listdir(datadir) if 'DS' not in d]
patterns = [d + f for d in dirs_toload\
            for f in ('*s1*STK','*s2*STK','*s3*STK' )]

savedir = '../data/'
for ldir in tqdm(patterns):
    # get name to save
    sdir = ldir.split('/')[-2] + '/'
    try:
        fname = glob.glob(ldir)[0].split('/')[-1].split('_w2')[0] + '_'+ldir.split('*')[-2]
    except IndexError:
        if 's1' in ldir:
            # probably not multi-stage movie, load single stage movie
            ldir = ldir.split('*')[0] + '*STK'
            fname = glob.glob(ldir)[0].split('/')[-1].split('_w2')[0]
            warnings.warn('loading {} as a single movie'.format(ldir))
        else: continue
    # create save directory if it doesn't exist yet
    try: mkdir(savedir+sdir)
    except FileExistsError: pass
    saveto = savedir + sdir + fname + '.tiff'
    print('processing {} and saving to {}'.format(ldir, saveto))
    load_zproject_STKcollection(ldir, savedir=saveto)
