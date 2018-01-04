"""
batch maximum intensity z-projection of movies
For movies taken with one or more stage positions
run as:
    python load_movs.py dir_to_load
where dir_to_load contains multiple directories, each with groups of 3 movies
if multistage movie not found, try loading single movie
"""
import numpy as np
from im_utils import z_project, load_zproject_STKcollection
from skimage import io
from sys import argv
from os import mkdir, listdir
import glob
from tqdm import tqdm
import warnings
import re

# get loading root and save directory from stdin or use default
try:
    datadir = argv[1]
    savedir = argv[2]
    if datadir[-1] != '/': datadir += '/'
    if savedir[-1] != '/': savedir += '/'
except IndexError:
    datadir = '/Volumes/PQdata/data/unloaded/'
    savedir = '../data/'
    print('No load nor save directories specified, using default.\
            \ndatadir: {0}\nsavedir: {1}'.format(datadir, savedir))

# get subdirectories with files to load
dirs = [datadir + d + '/' for d in listdir(datadir) if 'DS' not in d]

for _dir in dirs:
    # get path to stack files
    fnames_bydir = glob.glob(_dir + '*STK')
    # get stage positions
    stage_pos = list(set(re.search('s[0-9]+_', f)[0] + '*STK'\
                        for f in fnames_bydir))
    # get load patterns grouped by stage position
    patterns = [_dir + '*' + s for s in stage_pos]
    # get directory name
    sdir = _dir.split('/')[-2] + '/'
    # create save directory if it doesn't exist yet
    try: mkdir(savedir+sdir)
    except FileExistsError: pass

    for patt in tqdm(patterns):
        path_to_files = glob.glob(patt)
        fname = re.split('_w[1-9]', path_to_files[0].split('/')[-1])[0]
        saveto = savedir + sdir + fname + '.tif'
        print('processing {}'.format(fname))
        # make sure channels are in a consistent order (chronological order of
        # imaging, reverse alphabetical order): 650, 485, 387.
        path_to_files.sort(key=lambda x: x.split('-PENT-')[1], reverse=True)
        load_zproject_STKcollection(path_to_files, savedir=saveto)
