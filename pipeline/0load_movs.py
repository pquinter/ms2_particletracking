"""
batch maximum intensity z-projection of movies
run as:
    python load_movs.py dir_to_load
where dir_to_load contains multiple directories
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
from joblib import Parallel, delayed

datadir = '/Volumes/PQdata/data/PP7/2019/unloaded/'
savedir = '../data/2019_galinducedt0/'
dirs_toload = [datadir + d + '/' for d in listdir(datadir) if 'DS' not in d]
mov_names = [glob.glob(d+'*.nd')[0].split('/')[-1][:-3] for d in dirs_toload]
proj = Parallel(n_jobs=6)(delayed(load_zproject_STKcollection)
                (mov_dir+'*STK', savedir+name+'.tif')
                for mov_dir, name in tqdm(zip(dirs_toload, mov_names)))

