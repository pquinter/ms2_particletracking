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
for imname in tqdm(ims_projected_dapi):
    print('analyzing {}'.format(imname))
    im = ims_projected[imname]
    fish = im[:,:,0] # smFISH channel
    autof = im[:,:,1] # autofluorescent channel for cell segmentation
    dapi = im[:,:,2] # Dapi/nuclear channel

    # segment nuclei, channel with best signal ================================
    print('making nuclei mask with adaptive threshold...')
    mask_nuclei = mask_image(dapi, min_size=100, block_size=101,
                selem=skimage.morphology.disk(10))
    nuclei_markers = skimage.measure.label(mask_nuclei)
    print('finding nuclei centers...') # with peak finding algorithm
    nuc_centers = tp.locate(dapi, diameter=15, minmass=np.median(dapi)*50,
            separation=15)
    # convert center coordinates to image of nuclei seeds for watershed
    nuc_seeds = np.full_like(dapi, 0)
    for i, (_, r) in enumerate(nuc_centers.iterrows(), 1):
        nuc_seeds[int(r.y), int(r.x)] = i
    print('watershed reconstruction from centers...')
    # remove seeds not in nuclei mask and reconstruct with watershed
    nuc_seeds = nuc_seeds * mask_nuclei
    nuclei_markers = skimage.morphology.watershed(nuclei_markers,
                                    nuc_seeds, mask=mask_nuclei)

    # segment cells from nuclei markers =======================================
    cell_markers, nuclei_markers, mask_cells, mask_nuclei =\
                        segment_cellfromnuc(autof, nuclei_markers)
    markers_dict[imname] = cell_markers, nuclei_markers
    # draw segmented nuclei and cell boundaries on merged image
    ims_seg[imname] = make_seg_im((cell_markers, nuclei_markers),
        dapi + autof*3 + fish) # enhanced autof because its dimmer than others

    # make nuclei centers dataframe ===========================================
    # get centroids (so all bbox are same size)
    nuc_regionprops = skimage.measure.regionprops(nuclei_markers, dapi)
    _seg_coords = regionprops2df(nuc_regionprops, props=('label', 'centroid'))
    # add image name for sel_training func
    _seg_coords['imname'] = imname
    seg_coords = pd.concat((seg_coords, _seg_coords), ignore_index=True)
# expand centroid into x, y coords
seg_coords[['y','x']] = seg_coords.centroid.apply(pd.Series)
# add cell id
seg_coords['cid'] = seg_coords.apply(lambda x: x.imname+'_'+str(x.label), axis=1)

# pickle everything
with open('../output/nuc_trainingset/20171201_nuclei_segment_centroids_markers_segim.pkl', 'wb') as f:
    pickle.dump(seg_coords, f)
    pickle.dump(markers_dict, f)
    pickle.dump(ims_seg, f)
