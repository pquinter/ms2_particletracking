import pandas as pd
import glob
import re
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import particle

part_dir = '../output/pipeline_smfish/particles/parts_filtered.csv'
data_dir = '../../smFISH/data/TL47pQC7576/*/*/*tif'
impath_list = glob.glob(data_dir)

# filter out already processed movies, if any
try:
    parts_extant = pd.read_csv(part_dir)
    # get names of processed images
    processed_ims = parts_extant.mov_name.unique()
    # and exclude them from detection
    impath_list = [p for p in impath_list\
            if re.search(r'.+/(.+)(?:\.tif)$', p).group(1) not in processed_ims]
except FileNotFoundError: parts_extant = None

# Detect particles
parts = Parallel(n_jobs=12)(delayed(particle.locate_batch_smfish)(im_path)
        for im_path in tqdm(impath_list))
parts = pd.concat(parts, ignore_index=True)
parts = parts[parts.cell>0].reset_index(drop=True)

# update extant dataframe
if parts_extant is not None:
    parts = pd.concat((parts_extant, parts), sort=True, ignore_index=True)
parts.to_csv(part_dir.format(part_dir), index=False)
print('saved to {}'.format(part_dir))
