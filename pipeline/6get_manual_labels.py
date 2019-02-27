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
import im_utils

###############################################################################
# Identify spot labels
###############################################################################
# particle dataframe
part_dir = '../output/pipeline/particles'
parts = pd.read_csv(part_dir+'/parts_filtered.csv')
# sampled frames metadata
sampled_frames = pd.read_csv('../output/pipeline/sample_movs/sampled_frames.csv')
# Fiji manually annotated movies
mov_path = '../output/pipeline/sample_movs/labeled/labeled_sample_mov.tif'
mov_labeled = io.imread(mov_path)

# get manual labels
plabeled = []
for mov_name, _sampled_frames in sampled_frames.groupby('mov_name'):
    plabeled.append(particle.get_manual_labels(_sampled_frames, mov_name, mov_labeled, parts))
plabeled = pd.concat(plabeled, ignore_index=True)
plabeled.to_csv('{}/parts_labeled.csv'.format(part_dir), index=False)

###############################################################################
# Get labeled images
###############################################################################

spots_dir = '../output/pipeline/spot_images'
pids_all, rawims_all, bpims_all = particle.load_patches(spots_dir)
# check if x is in y for parallel elementwise comparison
el_isin = lambda x,y: x in y
# get which spots were in labeled movies
labeled = Parallel(n_jobs=12)(delayed(el_isin)(i, plabeled.pid.values)
                                            for i in tqdm(pids_all))
# filter out spots not present in labeled movies at all
pids_labeled = pids_all[labeled]
rawims_lbl, bpims_lbl = rawims_all[labeled], bpims_all[labeled]
# get PID of spots labeled as True and those not labeled
pids_true = np.isin(pids_labeled, plabeled[plabeled.manual_label].pid.values)
pids_false = ~pids_true

plot_dir = '../output/pipeline/plots'
# Plot labeled spots
for ims, name in zip((rawims_lbl, bpims_lbl), ('raw','bp')):
    # get images and save
    trueims, falseims = ims[pids_true], ims[pids_false]
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
fig = corner.corner(plabeled[~plabeled.manual_label][['corrwideal','mass_norm']], color='r', alpha=0.8, hist_kwargs={'density':True})
corner.corner(plabeled[plabeled.manual_label][['corrwideal','mass_norm']], color='b', fig=fig, hist_kwargs={'density':True})
plt.legend(['False','True'])
fig.axes[2].set(xlim=(-0.2,1), ylim=(4,12))
fig.axes[0].set(xlim=(-0.2,1), ylim=(0,4))
fig.axes[3].set(xlim=(3,12), ylim=(0,1))
sns.despine()
plt.tight_layout()
plt.savefig('{}/corner_labeled_spots.pdf'.format(plot_dir, name), bbox_inches='tight')

fig = corner.corner(parts[['corrwideal','mass_norm']], color='k', alpha=0.8, hist_kwargs={'density':True})
plt.legend([])
plt.savefig('{}/corner_all_spots.pdf'.format(plot_dir, name), bbox_inches='tight')
