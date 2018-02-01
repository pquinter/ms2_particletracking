import pandas as pd
import numpy as np
from collections import defaultdict

# load nuclei and particle tracking data
nuclei_peaks = pd.read_csv('../output/pp7/peaks_complete.csv')

# load z-projected movies
movs = {}
ddir = '../data/PP7/PP7_HJ_projected/'
for fname in tqdm(os.listdir(ddir)):
    if 'tif' in fname:
        _mov = io.imread(ddir + fname)
        _mname = fname.split('.')[0]
        movs[_mname] = _mov
    else: next

# make whole image tracking movies
track_movs = {}
for imname, group in tqdm(nuclei_peaks.groupby('imname')):
    track_movs[imname] = tracking_movie(movs[imname], group)

# Split by cell. Change 'coords_col' to xy and drop NaNs to split by particle
nuc_movs = defaultdict(list)
for (imname, label), nuc in nuclei_peaks.groupby(['imname', 'label']):
    nuc_movs[imname].append(get_batch_bbox(nuc, track_movs, size=50,
                                    movie=True, coords_col=['x_cell','y_cell']))

# make global nuclei tracking movie for easier visualization and save
norm=True
for m in nuc_movs:
    _movs = nuc_movs[m]
    if norm:
        _movs = [skimage.img_as_uint(normalize(_movs[i])) for i in range(len(_movs))]
    globmov = concat_movies(_movs, nrows=4)
    io.imsave('../output/pp7/movie_byCell{}.tif'.format(m), globmov)
