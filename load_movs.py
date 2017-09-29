"""
batch maximum intensity z-projection of movies
For movies taken with 3 different stage positions
run as:
    python load_movs.py dir_to_load
where dir_to_load contains multiple directories, each with groups of 3 movies
"""
import numpy as np
from im_utils import z_project
from skimage import io
from skimage.external.tifffile import TiffFile
from sys import argv
from os import mkdir, listdir
import glob
from tqdm import tqdm

# load movies and save them

def load_zproject_STKcollection(load_pattern, savedir=None):
    """
    Load collection of STK files and z-project each
    """
    collection = io.ImageCollection(load_pattern, load_func=TiffFile)
    collection = np.stack([z_project(zseries.asarray()) for zseries in collection])
    if savedir:
        io.imsave(savedir, collection)
    return collection

rootdir = argv[1] + '/'
datadir = '/Volumes/QuinDrive/Research/Data/YeastMicroscopy/' + rootdir
dirs_toload = [datadir + d + '/' for d in listdir(datadir) if 'DS' not in d]
patterns = [d + f for d in dirs_toload\
            for f in ('*s1*STK','*s2*STK','*s3*STK' )]

savedir = '../data/'
for ldir in tqdm(patterns):
    # get name to save
    sdir = ldir.split('/')[-2] + '/'
    fname = glob.glob(ldir)[0].split('/')[-1].split('_w2')[0] + '_'+ldir.split('*')[-2]
    # create save directory
    try: mkdir(savedir+sdir)
    except FileExistsError: pass
    saveto = savedir + sdir + fname + '.tiff'
    print('processing {} and saving to {}'.format(ldir, saveto))
    load_zproject_STKcollection(ldir, savedir=saveto)
