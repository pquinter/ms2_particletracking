import pandas as pd
import numpy as np
from collections import defaultdict


# load nuclei and particle tracking data
#nuclei_peaks = pd.read_pickle('../output/GFPEnvyScar/nuclei_peaks.p')

# load z-projected movies to make tracking movie
movs = {}
ddir = '../data/GFPEnvymScarlet/raw_projected/'
for fname in tqdm(os.listdir(ddir)):
    if 'tiff' in fname:
        movs[fname.split('.')[0]] = io.imread(ddir + fname)
    else: next

# get tracking movies of all nuclei for visualization
nuc_movs = defaultdict(list)
show_wholemov=True #show whole movie or just frames with id peaks
norm=True#normalize each frame to improve viz
for (name, label), nuc in nuclei_peaks.groupby(['imname', 'label']):
    if show_wholemov:
        # get whole movie from original
        b = nuc.bbox.iloc[0] # bounding box coordinates are the same for all frames
        bx, by = (slice(b[0], b[2]), slice(b[1], b[3]))
        nuc_mov = movs[name][:, bx, by]
    else:
        # frames with peaks are already in dataframe
        nuc_mov = np.stack(nuc.drop_duplicates('frame').bb_image)
    nuc_trackmov = tracking_movie(nuc_mov, nuc, x='bbx', y='bby')
    nuc_movs[name].append(nuc_trackmov)

# Movie by particle instead of nucleus
#nuclei_peaks = pd.read_pickle('../output/pp7/pp7spots_SVMfiltered_20171219.pkl')
nuclei_peaks = pd.read_pickle('../output/pp7/pp7spots_SVMfiltered_20171219.pkl')
nuc_movs = defaultdict(list)
show_wholemov=True #show whole movie or just frames with id peaks
norm=True#normalize each frame to improve viz
for (name, pid), nuc in nuclei_peaks.groupby(['imname', 'pid']):
    # get single set of bounding box coordinates for whole movie
    bx, by = get_bbox(nuc[['x','y']].iloc[0], size=50, return_im=False) # bounding box coordinates are the same for all frames
    nuc_mov = movs[name][:, bx, by]
    # transform particle whole movie coordinates to bounding box coordinates
    _coords = pd.DataFrame()
    _coords['frame'] = nuc.frame
    _coords['_bbx'] = nuc['x'] - by.start
    _coords['_bby'] = nuc['y'] - bx.start
    nuc_trackmov = tracking_movie(nuc_mov, _coords, x='_bbx', y='_bby')
    nuc_movs[name].append(nuc_trackmov)

# make global nuclei tracking movie for easier visualization and save
for m in nuc_movs:
    _movs = nuc_movs[m]
    if norm:
        _movs = [skimage.img_as_uint(normalize(_movs[i])) for i in range(len(_movs))]
    globmov = concat_movies(_movs, nrows=4)
    io.imsave('../output/pp7/tracking_movs_classif/movie_bypid_{}.tif'.format(m), globmov)
# visualize movie, probably better in .tif using fiji
#show_movie(globmov, 0.1, loop=True)
