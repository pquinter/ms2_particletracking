import pandas as pd
import numpy as np
from utils import particle
from joblib import delayed, Parallel
from tqdm import tqdm

part_dir = '../output/pipeline/particles/parts_filtered.csv'
savedir = '../output/pipeline/particles/parts_allframesimputed.csv'

parts = pd.read_csv(part_dir)
# forward-fill particle coordinates for missing frames
coords_complete = Parallel(n_jobs=12)(delayed(particle.impute_coords)(coords_df)
                        for _, coords_df in tqdm(parts.groupby(['roi','mov_name'])))
coords_complete = pd.concat(coords_complete, ignore_index=True)
# add the rest of the columns back; new rows get NaNs in all of these
coords_complete = pd.merge(coords_complete, parts,
                        on=list(coords_complete.columns), how='outer')
coords_complete.to_csv(savedir, index=False)
