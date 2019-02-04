"""
hexanediol treatment seems to work, eliminates bright particles formed with FUS and brings them back to WT
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from skimage import io
import skimage
from im_utils import *
import matplotlib.patches as mpatches

%matplotlib
mov = io.imread('/Users/porfirio/Desktop/08222018_pQC75/MAX_08222018_TL47pQC75_10%int100uLumen_HexT15_w1Brightfield_t1.TIF - GFPlow.tif')
mov = io.imread('/Users/porfirio/Desktop/09192018_pQC37vpQC6tl74_10%int150uCyan150msExp25uGreen200msExp_23minPosGal_w2GFPlow_proj.tif')
# project movie through time and apply median filter for better viz range
mov_filt = np.stack([skimage.filters.median(f) for f in mov])
mov = nohex75
mov_proj = skimage.filters.median(z_project(mov))

mov_proj = remove_cs(z_project(nohex75), perc=0.1)
cmap='viridis'

# get drift corrected movies of selected cells
cellmovs = []
# compute consensus shift
shifts=[]
for _roi in rois:
    shifts.append([drift_corr_roi(mov[0], mov[i], _roi) for i in range(0, len(mov))])
shifts = [np.median(s, axis=1).astype(int) for s in np.dstack(shifts)]
corr_rois = [[shift_roi(shift, roi) for shift in shifts] for roi in rois]
cellmovs = [np.stack([f[r] for f,r in zip(mov,_rois)]) for _rois in corr_rois]
test = concat_movies(cellmovs, ncols=len(cellmovs), norm=False)
# make kymograph
kymograph(test, vmin=None, remove_zeros=True)

# show selected cells
show_movie(concat_movies([mov_filt[:,r[0],r[1]] for r in rois], norm=False), delay=0.1)


zoom = (slice(1337, 1700, None), slice(256, 585, None))
# whole image
drift_corrected = np.fft.ifftn(fourier_shift(np.fft.fftn(hex75_3[-1]), shift)).real
mov_drifted = np.stack((hex75_3[0], hex75_3[-1]))
mov_corrected = np.stack(drift_corr(hex75_3[0], hex75_3[-1], roi=zoom))
mov_corrected = np.stack((ref, movie))
hex75_proj = skimage.filters.median(z_project(hex75_3))


hex75 = io.imread('/Users/porfirio/Desktop/08222018_pQC75/MAX_08222018_TL47pQC75_10%int100uLumen_HexT15_w1Brightfield_t1.TIF - GFPlow.tif')
hex75 = np.stack([skimage.filters.median(f) for f in hex75])
nohex75 = io.imread('/Users/porfirio/Desktop/08232018/MAX_08232018_TL47pQC75_5minPosGalNoHex_10%int100uLumCyan_w1Brightfield_t1.TIF - GFPlow.tif')
# last frame is for segmentation purposes only (100% intensity laser)
nohex75 = nohex75[:-1]
nohex75 = np.stack([skimage.filters.median(f) for f in nohex75])
fig, axes = plt.subplots(1, 2, sharey='row')
kymograph(hex75, ax=axes[0], vmin=100, vmax=600, cbar=False)
kymograph(nohex75, ax=axes[1], vmin=100, vmax=600, ylabel=False)
axes[0].axhline(31, ls='--', color='w', alpha=0.3)# hex addition
plt.tight_layout()
