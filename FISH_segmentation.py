from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import trackpy as tp
import pickle
from tqdm import tqdm
from im_utils import *
%matplotlib

# =============================================================================
# get nuclei centroid dataframe and nuclei markers for segmentation quality
# control =====================================================================

rrdir = '../data/FISH/20171201/'
ims_projected = load_ims(rrdir+'zprojected/', 'tif')
seg_coords, ims_seg , markers_dict = pd.DataFrame(), {}, {}
for imname in tqdm(ims_projected):
    print('analyzing {}'.format(imname))
    im = ims_projected[imname]
    fish = im[:,:,0] # smFISH channel
    autof = im[:,:,1] # autofluorescent channel for cell segmentation
    dapi = im[:,:,2] # Dapi/nuclear channel

    # segment nuclei, channel with best signal ================================
    print('finding nuclei centers...') # with peak finding algorithm
    nuc_centers = tp.locate(dapi, diameter=15, minmass=np.median(dapi)*50,
            separation=15)
    # convert center coordinates to image of nuclei seeds for watershed
    nuc_seeds = np.full_like(dapi, 0) # zero filled array
    for i, (_, r) in enumerate(nuc_centers.iterrows(), 1):
        nuc_seeds[int(r.y), int(r.x)] = i # add labeled centers to array

    print('nuclei watershed reconstruction from centers...')
    nuclei_markers, center_markers = segment_from_seeds(dapi, nuc_seeds,
            (100, 101, 10), dilate=False)

    print('cell watershed reconstruction from nuclei...')
    cell_markers, nuclei_markers = segment_from_seeds(autof, nuclei_markers,
            (1000, 151, 15), dilate=True)

    markers_dict[imname] = cell_markers, nuclei_markers
    # draw segmented nuclei and cell boundaries on merged image
    ims_seg[imname] = make_seg_im((cell_markers, nuclei_markers),
        dapi + autof*3 + fish) # enhanced autof because its dimmer than others

    # update nuclei centers dataframe ===========================================
    nuc_centers['label'] = nuc_centers.apply(lambda coords:\
            nuclei_markers[int(coords.y), int(coords.x)], axis=1)
    _seg_coords = nuc_centers[(nuc_centers.label>0)].reset_index(drop=True)
    # add image name for sel_training func
    _seg_coords['imname'] = imname
    seg_coords = pd.concat((seg_coords, _seg_coords), ignore_index=True)
# add cell id
seg_coords['cid'] = seg_coords.apply(lambda x: x.imname+'_'+str(x.label), axis=1)

# pickle everything
with open('../output/pipeline/20171201_nuclei_segment_centroids_markers_segim.pkl', 'wb') as f:
    pickle.dump(seg_coords, f)
    pickle.dump(markers_dict, f)
    pickle.dump(ims_seg, f)
