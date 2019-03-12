import pandas as pd
import glob
from utils import particle
from joblib import delayed, Parallel
from tqdm import tqdm

part_dir = '../output/pipeline_snapshots/particles/parts_filtered.csv'
data_dir = '../data/2019_pp7Snapshots/*/*tif'
spots_path = '../output/pipeline_snapshots/spot_images'

movpath_list = glob.glob(data_dir)
parts = pd.read_csv(part_dir)
# Process movies in parallel using 3 cores
# can go up to 6 cores, but each iteration needs a ton of memory
Parallel(n_jobs=6)(delayed(particle.get_patches)(mov_path, spots_path, parts, movie=False)
                    for mov_path in tqdm(movpath_list))
