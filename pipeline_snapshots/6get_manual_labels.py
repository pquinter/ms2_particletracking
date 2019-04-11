from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from skimage import io
import pickle
import glob
from utils import particle
import corner
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import im_utils

###############################################################################
# Identify spot labels
###############################################################################
# particle dataframe
part_dir = '../output/pipeline_snapshots/particles'
parts = pd.read_csv(part_dir+'/parts_filtered.csv')
# Fiji manually annotated movies
training_dir = '../output/pipeline_snapshots/GPClassification/training/'
im_path = glob.glob(training_dir+'labeled/*.tif')

def get_manual_label_im(lbld_impath, parts, brush_val=0):
    im_name = lbld_impath.split('/')[-1][:-4]
    parts_im = parts[parts.mov_name==im_name].copy()
    im_labeled = io.imread(lbld_impath)
    parts_im['int_value'] = parts_im.apply(lambda coords:\
                    im_labeled[int(coords.y), int(coords.x)], axis=1)
    # get label (spots were labeled in Fiji, by drawing a paintbrush)
    parts_im['manual_label'] = parts_im.int_value.values == brush_val
    del parts_im['int_value']
    return parts_im

# get manual labels
parts_l = []
for p in im_path:
    parts_l.append(get_manual_label_im(p, parts))
plabeled = pd.concat(parts_l, ignore_index=True)

plabeled.to_csv('{}/parts_labeled.csv'.format(training_dir), index=False)

###############################################################################
# Get labeled images
###############################################################################

spots_dir = '../output/pipeline_snapshots/spot_images'
pids_all, rawims_all, bpims_all = particle.load_patches(spots_dir)
# check if x is in y for parallel elementwise comparison
el_isin = lambda x,y: x in y
# complexity is O(1) with sets, O(n) with lists!!!
plabeled_set = set(plabeled.pid.values)
labeled = [i in plabeled_set for i in tqdm(pids_all)]
# filter out spots not present in labeled movies at all
pids_labeled = pids_all[labeled]
rawims_lbl, bpims_lbl = rawims_all[labeled], bpims_all[labeled]
# get PID of spots labeled as True and those not labeled
pids_true = np.isin(pids_labeled, plabeled[plabeled.manual_label].pid.values)
pids_false = ~pids_true

plot_dir = '../output/pipeline_snapshots/GPClassification/training'
# Plot labeled spots
for ims, name in zip((rawims_lbl, bpims_lbl), ('raw','bp')):
    # get images and save
    trueims, falseims = ims[pids_true], ims[pids_false]
    try: os.mkdir('{}/labeled/'.format(spots_dir))
    except FileExistsError: pass
    with open('{}/labeled/labeledspotims_15x15_{}.p'.format(spots_dir, name), 'wb') as f:
        pickle.dump(trueims, f)
        pickle.dump(falseims, f)
    # plot sorted by intensity
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(im_utils.im_block(trueims, 50, norm=False, sort=np.max), cmap='gray')
    axes[0].set_title('Labeled as spots')
    axes[1].imshow(im_utils.im_block(falseims, 80, norm=False, sort=np.max), cmap='gray')
    axes[1].set_title('Not labeled')
    [ax.set(xticks=[], yticks=[]) for ax in axes]
    plt.tight_layout()
    plt.savefig('{}/labeled_spots_{}.png'.format(plot_dir, name), bbox_inches='tight', dpi=300)
# Plot distribution of normalized mass and correlation with ideal spot
fig = corner.corner(plabeled[~plabeled.manual_label][['corrwideal','mass']], color='r', alpha=0.8, hist_kwargs={'density':True}, plot_contours=False)
corner.corner(plabeled[plabeled.manual_label][['corrwideal','mass']], color='b', fig=fig, hist_kwargs={'density':True}, plot_contours=False)
plt.legend(['False','True'])
ylim = plabeled.mass.max()
fig.axes[2].set(xlim=(-0.2,1), ylim=(0,ylim))
fig.axes[0].set(xlim=(-0.2,1), ylim=(0,8))
fig.axes[3].set(xlim=(3,ylim), ylim=(0,0.01))
sns.despine()
plt.tight_layout()
plt.savefig('{}/corner_labeled_spots.pdf'.format(plot_dir, name), bbox_inches='tight')

fig = corner.corner(parts[['corrwideal','mass']], color='k', alpha=0.8, hist_kwargs={'density':True})
plt.legend([])
plt.savefig('{}/corner_all_spots.pdf'.format(plot_dir, name), bbox_inches='tight')
