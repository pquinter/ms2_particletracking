"""
Generate sample movies for manual particle detection
"""
import pandas as pd
import numpy as np
import skimage
from skimage import io
import im_utils

import glob
from tqdm import tqdm
import im_utils

# path to movies, spot dataframe and masks
part_dir = '../output/pipeline/particles/parts_filtered.csv'
data_dir = '../data/2019_galinducedt0/*tif'
parts = pd.read_csv(part_dir)
movpath_list = glob.glob(data_dir)

# dataframe to store sampled frame numbers
sampled_frames = pd.DataFrame()

# sample size (length of movie to make with random frame sample)
ssize = 5
s_mov = []
for mov_path in tqdm(movpath_list):
    print('Analyzing {0}...'.format(mov_path))
    mov_name = mov_path.split('/')[-1][:-4]
    if mov_name not in parts.mov_name.unique():
        print('bad movie, skipping...')
        continue
    mov = io.imread(mov_path)
    # get random sample indices from frame 30 onwards (when first spots appear)
    np.random.seed(42)
    f_sample = np.sort(np.random.choice(np.arange(30, len(mov)),
        size=ssize, replace=False))
    print('getting sample movie...')
    s_mov.extend(mov[f_sample])
    # save sampled frames indices and movie name
    _sampled = pd.DataFrame()
    _sampled['frame'] = f_sample
    _sampled['mov_name'] = mov_name
    sampled_frames = pd.concat((sampled_frames, _sampled)).reset_index(drop=True)
    print('done')
s_mov = np.stack(s_mov)
io.imsave('../output/pipeline/sample_movs/sample_mov.tif', s_mov)
sampled_frames.to_csv('../output/pipeline/sample_movs/sampled_frames.csv', index=False)
