from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import skimage
from skimage import io
from im_utils import *
import scipy
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import trackpy as tp

##############################################################################
# Drift correction
##############################################################################

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
    drift_corrected_fourier = scipy.ndimage.fourier_shift(np.fft.fftn(drifted), shift)
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

##############################################################################
# operations of fluorescence traces
##############################################################################

def smooth_traces(traces, smooth_win=3, smooth_func=np.mean):
    return pd.DataFrame(traces).rolling(smooth_win, axis=1).apply(smooth_func, raw=True).dropna(axis=1).values

def get_traces(movies, top_px=5, norm=True, plot=True, ax=None, cmap='viridis',
        pad=None, smooth=False, **smooth_args):
    """
    Get pp7 traces from movies
    top_px: int
        number of top pixels to get median from
    norm: bool
        normalize traces by mean of each frame
    plot: bool
        plot traces as image
    """
    if norm:
        traces = np.array([[np.mean(np.sort(f.ravel())[-top_px:])/np.mean(f) for f in m] for m in movies])
    else:
        traces = np.array([[np.mean(np.sort(f.ravel())[-top_px:]) for f in m] for m in movies])
    if pad:
        traces = np.pad(traces, ((0,0), (pad,pad)), 'mean')
    if smooth:
        traces = smooth_traces(traces, **smooth_args)
    if plot:
        if ax is None: fig, ax = plt.subplots()
        ax.imshow(traces, cmap=cmap)
    return traces

def get_peak_traces(traces, peaks, win=10):
    """
    get individual peak fluor traces +-win frames around peak center
    traces: array
    peaks: peaks object from scipy.signal.find_peaks
    win: int
    """
    # pad traces first to make sure peaks are centered
    traces = np.pad(traces, ((0,0), (win,win)), 'mean')
    peak_traces = []
    for ps, _trace in zip(peaks, traces):
        # now need to correct peak centers for padding
        [peak_traces.append(_trace[win+p_center-10:win+p_center+11]) for p_center in ps[0]]
    return np.vstack(peak_traces)

#def correlate(a,b,i=1):
    # experimental implementation of Coulon and Larson (2016)
    # not working
#    N = len(a)
#    R = 0
#    for q in range(N-i-1):
#        R += a[q] * b[q+i]
#    return R/(N-i)

