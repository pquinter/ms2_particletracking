"""
Segment 3-channel z-projected smFISH images
"""
import glob
from joblib import Parallel, delayed
from skimage import io
import skimage.filters
from tqdm import tqdm
from utils import image
import os

datadir = '../../smFISH/data/TL47pQC7576/'
seg_dir = '../output/pipeline_smfish/segmentation/'
dirs_toload = [datadir + d + '/' for d in os.listdir(datadir) if 'DS' not in d]
path_list = [glob.glob(ddir+'/*/*tif') for ddir in dirs_toload]
path_list = [p for group in path_list for p in group]
# Segment cells (skips already segmented movies)
segmentation = Parallel(n_jobs=12)(delayed(image.segment_image_smfish)(impath, seg_dir)
        for impath in tqdm(path_list))
