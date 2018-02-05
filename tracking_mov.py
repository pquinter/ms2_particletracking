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

# load segmentation movies
seg_output_dir = '../output/pp7/seg_movs.p'
with open(seg_output_dir, 'rb') as f:
    seg_movs = pickle.load(f)
    seg_movs_proj = pickle.load(f)

# make whole image tracking movies
track_movs = {}
for imname, group in tqdm(nuclei_peaks.groupby('imname')):
    track_movs[imname] = tracking_movie(seg_movs[imname], group)

# Split by cell. Change 'coords_col' to xy and drop NaNs to split by particle
nuc_movs = defaultdict(list)
for (imname, label), nuc in nuclei_peaks.groupby(['imname', 'label']):
    nuc_movs[imname].append(get_batch_bbox(nuc, track_movs, size=50,
                                    movie=True, coords_col=['x_cell','y_cell']))

# make global nuclei tracking movie for easier visualization and save
for m in nuc_movs:
    _movs = nuc_movs[m]
    globmov = concat_movies(_movs, ncols=8, norm=1)
    io.imsave('../output/pp7/segmovie_byCell{}.tif'.format(m), skimage.img_as_int(globmov))

# manually select cells to trash
# get rid of those with no identified particles
movs4sel = []
peaks4sel = nuclei_peaks.dropna(subset=['particle'])
for (imname, label), nuc in peaks4sel.groupby(['imname', 'label']):
    movs4sel.append(get_batch_bbox(nuc, track_movs, size=50,
                                movie=True, coords_col=['x_cell','y_cell']))
globmov = concat_movies(movs4sel, ncols=20, norm=1)
io.imsave('../output/pp7/ref_segmovies.tif', skimage.img_as_int(globmov))

# keep only one frame for selection
peaks4sel = peaks4sel.drop_duplicates(subset=['label','imname']).sort_values(['imname','label'])
sel_ind, ims, peaks4sel = sel_training(peaks4sel, seg_movs_proj, s=50,
                                    ncols=20, coords_col=['x_cell','y_cell'])
sel_cpid = peaks4sel[sel_ind].cpid.values
# get rid of those with no identified particles
sel_peaks = nuclei_peaks[nuclei_peaks.cpid.isin(sel_cpid)]
with open('../output/pp7/05Feb2018sel_movs.p', 'wb') as f:
     pickle.dump(sel_peaks, f)

selmovs = []
for (imname, label), nuc in sel_peaks.groupby(['imname', 'label']):
    selmovs.append(get_batch_bbox(nuc, track_movs, size=50,
                                movie=True, coords_col=['x_cell','y_cell']))
globmovsel = concat_movies(selmovs, ncols=20, norm=1)
io.imsave('../output/pp7/ref_segmovies_selected.tif', skimage.img_as_int(globmov))
