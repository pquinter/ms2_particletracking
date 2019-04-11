import pandas as pd
import glob
import re
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import particle

part_dir = '../output/pipeline_snapshots/particles/parts_filtered.csv'
data_dir = '../data/2019_pp7Snapshots/*/*tif'
movpath_list = glob.glob(data_dir)

# filter out already processed movies, if any
try:
    parts_extant = pd.read_csv(part_dir)
    # get names of processed movies
    processed_movs = parts_extant.mov_name.unique()
    # and exclude them from detection
    movpath_list = [p for p in movpath_list\
            if re.search(r'.+/(.+)(?:\.tif)$', p).group(1) not in processed_movs]
except FileNotFoundError: parts_extant = None

# Detect particles
parts = Parallel(n_jobs=12)(delayed(particle.locate_batch)(mov_path, movie=False)
            for mov_path in tqdm(movpath_list))
# Cleanup based on segmented ROIs (see cleanup_track func for other criteria)
parts_filt = Parallel(n_jobs=12)(delayed(particle.cleanup_track)(_parts, remove_hotpixels=False, traj_len_thresh=0)
            for _parts in tqdm(parts))
parts_filt = pd.concat(parts_filt, ignore_index=True)

# update extant dataframe
if parts_extant is not None:
    parts_filt = pd.concat((parts_extant, parts_filt), sort=True, ignore_index=True)
parts_filt.to_csv(part_dir.format(part_dir), index=False)
print('saved to {}'.format(part_dir))
#
#im = io.imread(movpath_list[-1])
#plt.figure()
#im_bp = tp.bandpass(im, 1, 5)
#test = tp.locate(im, diameter=3, **locate_kwargs)
#fig, ax = plt.subplots(2, sharex=True, sharey=True)
#tp.annotate(test[test.mass>test.mass.mean()*10], im, ax=ax[0])
#ax[1].imshow(im)
