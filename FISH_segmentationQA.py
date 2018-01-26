from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from skimage import io
import pickle
from im_utils import *

input_dir = '../output/pipeline/20171201_nuclei_segment_centroids_markers_segim.pkl'
output_dir = '../output/nuc_trainingset/20171201_nuclei_segment_centroids_markers_segim.pkl'
seg_output_dir = '../output/20171201/segmentation/'
# =============================================================================
# Manually clean cell and nuclei markers
# =============================================================================
with open(input_dir, 'rb') as f:
    seg_coords = pickle.load(f)
    markers_dict = pickle.load(f)
    ims_seg_dict = pickle.load(f)

# Select crap markers
sel_ind, seg_ims, seg_coords = sel_training(seg_coords, ims_proj, ncols=200,
        normall=1, mark_center=0, s=9, cmap='viridis', step=200)
seg_coords['is_nucleus'] = ~sel_ind
#plt.imshow(im_block(np.concatenate(seg_ims)[sel_ind], 5, norm=1))

# Remove crap markers and update segmentation
sel_markers_dict = {}
for imname, _seg_coords in seg_coords.groupby('imname'):
    cell_markers, nuclei_markers = markers_dict[imname]
    for crap_label in _seg_coords[_seg_coords.is_nucleus==False].label.values:
        # remove crap from arrays of cell and nuclei markers
        cell_markers[np.where(np.isclose(cell_markers,crap_label))] = 0
        nuclei_markers[np.where(np.isclose(nuclei_markers,crap_label ))] = 0
    sel_markers_dict[imname] = cell_markers, nuclei_markers
    # make new segmentation image and save
    seg_im = make_seg_im((cell_markers, nuclei_markers), ims_seg_dict[imname])
    io.imsave(seg_output_dir + '{}_segmentation.tif'.format(imname),
                                        skimage.img_as_int(seg_im))
with open(output_dir, 'wb') as f:
    pickle.dump(seg_coords, f)
    pickle.dump(sel_markers_dict, f)
