from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from skimage import io
import pickle
from im_utils import *

# =============================================================================
# Manually clean cell and nuclei markers
# =============================================================================
with open('../output/pipeline/20171201_nuclei_segment_centroids_markers_segim.pkl', 'rb') as f:
    seg_coords = pickle.load(f)
    markers_dict = pickle.load(f)
    ims_seg_dict = pickle.load(f)

# Select crap markers
sel_ind, seg_ims, seg_coords_ = [], [], pd.DataFrame()
step = 200
for n in np.arange(step, len(seg_coords)+step, step):
    _ind, _ims, _seg_coords = sel_training(seg_coords[n-step:n], ims_seg_dict,
                        s=100, ncols=20, figsize=(25.6, 13.6), normall=1)
    sel_ind.append(_ind)
    seg_ims.append(_ims)
    seg_coords_ = pd.concat((seg_coords_, _seg_coords), ignore_index=True)
sel_ind = np.concatenate(sel_ind)
seg_coords_['is_nucleus'] = ~sel_ind
#plt.imshow(im_block(np.concatenate(seg_ims)[sel_ind], 5, norm=1))

# Remove crap markers and update segmentation
sel_markers_dict = {}
for imname, _seg_coords in seg_coords_.groupby('imname'):
    cell_markers, nuclei_markers = markers_dict[imname]
    for crap_label in _seg_coords[_seg_coords.is_nucleus==False].label.values:
        cell_markers[np.where(np.isclose(cell_markers,crap_label))] = 0
        nuclei_markers[np.where(np.isclose(nuclei_markers,crap_label ))] = 0
    sel_markers_dict[imname] = cell_markers, nuclei_markers
    # make new segmentation image and save
    seg_im = make_seg_im((cell_markers, nuclei_markers), ims_fishdapiautof_merge[imname])
    io.imsave('../output/segmentation/{}_segmentation.tif'.format(imname),
                                        skimage.img_as_int(seg_im))
with open('../output/nuc_trainingset/20171201_nuclei_segment_centroids_markers_segim.pkl', 'wb') as f:
    pickle.dump(seg_coords, f)
    pickle.dump(sel_markers_dict, f)
