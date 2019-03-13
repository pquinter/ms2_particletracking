import os
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.image import load_zproject_STKimcollection

datadir = '/Volumes/PQdata/data/PP7/032019_PP7SnapShots/'
savedir = '../data/2019_pp7Snapshots/'
dirs_toload = [datadir + d + '/' for d in os.listdir(datadir) if 'DS' not in d]
proj = Parallel(n_jobs=6)(delayed(load_zproject_STKimcollection)
                (im_dir+'*STK', savedir) for im_dir in tqdm(dirs_toload))
