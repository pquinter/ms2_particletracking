import pandas as pd
import numpy as np
import pickle
import glob
from utils import image, particle
import im_utils

spots_dir = '../output/pipeline_snapshots/spot_images'
part_dir = '../output/pipeline_snapshots/particles/parts_filtered.csv'

pids_all, rawims_all, bpims_all = particle.load_patches(spots_dir)
# compute correlation with ideal spot
win9x9 = np.s_[:,3:12,3:12]
corrs = im_utils.corr_widealspot(rawims_all[win9x9], wsize=9, PSFwidth=4.2)
# put in dataframe and merge
corrs_df = pd.DataFrame({'corrwideal':corrs, 'pid':pids_all})
parts = pd.read_csv(part_dir)
parts = pd.merge(parts, corrs_df, on='pid')
parts.to_csv(part_dir, index=False)
