"""
Batch maximum intensity z-projection and copy FISH channel 3D stacks
run as:
    python load_movs.py dir_to_load dir_to_save
where dir_to_load contains multiple directories, each with groups of 3 movies
Saving according to the following directory structure:
    |--savedir
        |--ExperimentDate1 (YYYYMMDD e.g. 20171224)
            |--Strain1 (YYYYMMDD_Strain#Line# e.g. 20171224_666)
                |---Date_Strain1_1.tif
                |---Date_Strain1_2.tif...
            |--Strain2 ...
        |--ExperimentDate2 ...
"""
from im_utils import load_zproject_STKcollection
import argparse
import os
import glob
import re
from joblib import Parallel, delayed
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help="""Data directory
                    e.g. ''/Volumes/PQdata/data/FISH/TL47_pQC7576''
                    Contains subdirectories with the following structure:
                    date --> strain --> images """)
parser.add_argument('save_dir', help="""Saving directory for projected images
                                        e.g. '../data/TL47_pQC7576'""")
parser.add_argument('ext', help="""Extension of files to project e.g. 'STK'""")
args = parser.parse_args()
datadir = args.data_dir
savedir = args.save_dir
ext = args.ext
if datadir[-1] != '/': datadir += '/'
if savedir[-1] != '/': savedir += '/'

# get subdirectories with files to load
dirs = [datadir + d + '/' for d in os.listdir(datadir) if 'DS' not in d]
# create experiment data directory
os.mkdir(savedir)

for _dir in tqdm(dirs):
    # get date
    date = re.match('.*(\d{8}).*', _dir).group(1)
    # make date directories
    os.mkdir(savedir+date)
    # get strains and respective directories
    strain_dirs = [(re.match('\W*([TL|yQC]\w+)\W*', d).group(1), d)\
                            for d in os.listdir(_dir) if 'DS' not in d]
    for (strain, sdir) in strain_dirs:
        # create saving directory
        zproj_dir = savedir+date+'/'+strain+'/'
        os.mkdir(zproj_dir)
        # get path to stack files
        stack_path = glob.glob(_dir+sdir+'/*{}'.format(ext))
        # get patterns to load three channels
        patterns = [d.split('w1')[0] + '*'  for d in stack_path]

        def getsavedir(p, date, strain):
            # get image number
            im_no = re.match('.+_([1-9][0-9]?)_.+', p).group(1)
            # put save directory together
            saveto = zproj_dir+'{}_{}_{}.tif'.format(date,strain,im_no)
            return saveto

        projected = Parallel(n_jobs=12)(delayed(load_zproject_STKcollection)
                        (p, getsavedir(p, date, strain)) for p in tqdm(patterns, desc='z-projecting'))
