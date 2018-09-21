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
def drift_corr(ref, drifted, roi=slice(None)):
    """
    drift correction with FFT autocorr.
    Histogram is modified with fourier shift, need to be careful
    """
    from skimage.feature import register_translation
    ref_roi = ref[roi]
    drifted_roi = drifted[roi]
    # find XY shift with subpixel precision
    shift, error, diffphase = register_translation(ref_roi, drifted_roi, 100)
    # shift back in fourier space, then invert fourier
    drift_corrected_fourier = fourier_shift(np.fft.fftn(drifted), shift)
    drift_corrected = np.fft.ifftn(drift_corrected_fourier).real
    return ref, drift_corrected

def drift_corr_roi(ref, drifted, roi):
    """
    Compute shift for drift correction with FFT autocorr. of roi object
    """
    from skimage.feature import register_translation
    ref_roi = ref[roi]
    drifted_roi = drifted[roi]
    # find XY shift
    shift, error, diffphase = register_translation(ref_roi, drifted_roi)
    return shift

def shift_roi(shift, roi):
    # shift roi
    corr_roi = [slice(r.start-int(s), r.stop-int(s)) for r,s in zip(roi, shift)]
    return corr_roi

mov = io.imread('/Users/porfirio/Desktop/08222018_pQC75/MAX_08222018_TL47pQC75_10%int100uLumen_HexT15_w1Brightfield_t1.TIF - GFPlow.tif')
mov = io.imread('/Users/porfirio/Desktop/09192018_pQC37vpQC6tl74_10%int150uCyan150msExp25uGreen200msExp_23minPosGal_w2GFPlow_proj.tif')
# project movie through time and apply median filter for better viz range
mov_filt = np.stack([skimage.filters.median(f) for f in mov])
mov = nohex75
mov_proj = skimage.filters.median(z_project(mov))

mov_proj = remove_cs(z_project(nohex75), perc=0.1)
cmap='viridis'
# Manual selection of active cells

def drawROIedge(roi, im, lw=2, fill_val='max'):
    """
    Draw edges around Region of Interest in image

    Arguments
    ---------
    roi: tuple of slice objects, as obtained from zoom2roi
    im: image that contains roi
    lw: int
        edge thickness to draw
    fill_val: int
        intensity value for edge to draw

    Returns
    --------
    _im: copy of image with edge around roi

    """
    _im = im.copy()
    # get value to use for edge
    if fill_val=='max': fill_val = np.max(_im)
    # get start and end of rectangle
    x_start, x_end = roi[0].start, roi[0].stop
    y_start, y_end = roi[1].start, roi[1].stop
    # draw it
    _im[x_start:x_start+lw, y_start:y_end] = fill_val
    _im[x_end:x_end+lw, y_start:y_end] = fill_val
    _im[x_start:x_end, y_start:y_start+lw] = fill_val
    _im[x_start:x_end, y_end:y_end+lw] = fill_val
    return _im

def manual_roi_sel(mov, rois=None):
    """
    Manuallly crop multiple regions of interest (ROI)
    from max int projection of movie

    Arguments
    ---------
    mov: array like
        movie to select ROIs from
    rois: list, optional
        list of coordinates, to be updated with new selection

    Returns
    ---------
    roi_movs: array_like
        list of ROI movies
    rois: list
        updated list of ROI coordinates, can be used as frame[coords]

    """
    mov_proj = skimage.filters.median(z_project(mov))
    if not rois: rois = []
    # max projection of movie to single frame
    _mov_proj = mov_proj.copy()
    fig, ax = plt.subplots(1)
    ax.set(xticks=[], yticks=[])
    while True:
        plt.tight_layout() # this fixes erratic drawing of ROI contour
        ax.set_title('zoom into roi, alt+click, then click again to add\nclose figure to finish',
                fontdict={'fontsize':12})
        ax.imshow(_mov_proj, cmap=cmap)
        _ = plt.ginput(10000, timeout=0, show_clicks=True)
        roi = zoom2roi(ax)
        # check if roi was selected, or is just the full image
        if roi[0].start == 0 and roi[1].stop == _mov_proj.shape[0]-1:
            plt.close()
            break
        else:
            # mark selected roi and save
            _mov_proj = drawROIedge(roi, _mov_proj)
            rois.append(roi)
            # zoom out again
            plt.xlim(0, _mov_proj.shape[1]-1)
            plt.ylim(_mov_proj.shape[0]-1,0)
    roi_movs = [np.stack([f[r] for f in mov]) for r in rois]
    return roi_movs, rois


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
