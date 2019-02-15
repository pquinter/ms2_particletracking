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

##############################################################################
# Manual cell selection
##############################################################################

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
    # check if multiple rois
    if not isinstance(roi[0], slice):
        for r in roi:
            _im = drawROIedge(r, _im)
        return _im
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

def manual_roi_sel(mov, rois=None, cmap='viridis', plot_lim=5):
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
    # max projection of movie to single frame
    mov_proj = skimage.filters.median(z_project(mov))
    mov_proj = mov_proj.copy()
    mov_proj = remove_cs(mov_proj, perc=0.001, tol=2)
    if rois: mov_proj = drawROIedge(rois, mov_proj)
    else: rois = []
    fig, ax = plt.subplots(1)
    ax.set(xticks=[], yticks=[])
    i=0
    while True:
        # make new figure to avoid excess memory consumption
        if i>plot_lim:
            plt.close('all')
            fig, ax = plt.subplots(1)
            ax.set(xticks=[], yticks=[])
        plt.tight_layout() # this fixes erratic drawing of ROI contour
        ax.set_title('zoom into roi, press Enter, then click again to add\nPress Enter to finish',
                fontdict={'fontsize':12})
        ax.imshow(mov_proj, cmap=cmap)
        _ = plt.ginput(10000, timeout=0, show_clicks=True)
        roi = zoom2roi(ax)
        # check if roi was selected, or is just the full image
        if roi[0].start == 0 and roi[1].stop == mov_proj.shape[0]-1:
            plt.close()
            break
        else:
            # mark selected roi and save
            mov_proj = drawROIedge(roi, mov_proj)
            rois.append(roi)
            # zoom out again
            plt.xlim(0, mov_proj.shape[1]-1)
            plt.ylim(mov_proj.shape[0]-1,0)
        i+=1
    roi_movs = [np.stack([f[r] for f in mov]) for r in rois]
    return roi_movs, rois

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

##############################################################################
# Plotting
##############################################################################
def kymograph(movies, ax=None, t=20, vmin=100, vmax=750, ylabel=True, title=None,
        xticks=0, log=False, cbar=True, plot=True, remove_zeros=False,
        cosmic_safe=True, cmap='viridis'):
    """ Make and plot kymograph of movie """
    if ax is None:
        fig, ax = plt.subplots()
    # remove cosmic rays
    if cosmic_safe:
        mov = [remove_cs(m) for m in movies]
    # concatenate movies if more than one
    if isinstance(movies, list) or movies.ndim>3:
        mov = concat_movies(mov, ncols=len(movies), norm=False)
    # max intensity projection along columns; x coord is same in mov as in plot
    rowmax = pd.DataFrame([np.max(f, axis=0) for f in mov])
    # convert frames to minutes
    rowmax.index = [int(i) for i in rowmax.index*t/60]
    if log:
        rowmax = np.log(rowmax)
        vmin, vmax = np.min(rowmax.values), np.max(rowmax.values)
    if remove_zeros:
        # remove all zero columns, useful for kymograph of concatenated movies
        rowmax = rowmax[rowmax>0].dropna(how='all', axis=1)
    if plot:
        sns.heatmap(rowmax, ax=ax, cmap=cmap, xticklabels=xticks,
            yticklabels=8, vmin=vmin, vmax=vmax, cbar=cbar)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
        if ylabel: ax.set_ylabel('Time (min)')
        if title: ax.set_title(title)
        return rowmax, fig, ax
    else: return rowmax

def shade_peaks_stackp(axes, peaks, traces, t=20, scolor='w', salpha=0.3, pcolor='#FD7E96'):
    """
    Shade detected peaks on stackplot

    peaks: list of peaks
        peaks objects from scipy.signal.find_peaks
    traces: arrays
        fluorescence traces
    t: int
        time interval in seconds
    scolor: str
        shade color
    salpha: float
        shade alpha
    pcolor: str
        color for vertical line at peak
    """

    # make time array for each trace
    time_arr = [np.arange(t, 1+len(tr)*t, t)/60 for tr in traces]
    for ax, ps, time, _trace in zip(axes, peaks, time_arr, traces):
        left = ps[1]['left_bases']
        right = ps[1]['right_bases']
        # shade each peak
        [ax.fill_between(np.linspace(time[l], time[r], len(_trace[l:r])),
            _trace[l:r], color=scolor, alpha=salpha) for l, r in zip(left,right)]
        # plot vertical line at peak
        if pcolor:
            [ax.axvline(time[p], ymax=0.5, color=pcolor) for p in ps[0]]
    return None

def stackplot(traces, t=20, hspace=0.6, c='#00667A', oc='w',
        shade_peaks=None, **shade_peaks_kwargs):
    """
    Create a stackplot with traces
    matplotlib's out-of-the-box stackplot looks confusing; do manual here
    t: int
        time interval in seconds
    hspace: float [0-1]
        how tightly to stack vertically: 0 is well separated, -1 overlapping
        can also be adjusted after with: `fig.subplots_adjust(hspace=-0.1)`
    c: area color
    oc: outline color
    """
    # make time array for each trace in minutes (can be of different lengths)
    time_arr = [np.arange(t, 1+len(tr)*t, t)/60 for tr in traces]
    fig, axes = plt.subplots(len(traces), sharex=True, sharey=False)
    axes = axes.ravel()
    # plot area
    [ax.fill_between(time, tr, color=c)\
            for ax, time, tr in zip(axes, time_arr, traces)]
    # plot lighter outline
    [ax.plot(time, tr, c=oc, alpha=0.5, ls='-', lw=2)\
            for ax, time, tr in zip(axes, time_arr, traces)]
    # make background transparent
    [ax.patch.set_visible(False) for ax in axes]
    # get x and y limits
    max_y = np.max([np.max(a) for a in traces])
    min_y = int(np.min([np.min(a) for a in traces]))*0.5 #50% of min y for viz
    max_x = np.max([a[-1] for a in time_arr])
    min_x = t/60

    # set max y tick in upper ax, min y and xlabel in lower ax
    axes[0].set(yticks=(max_y,))
    axes[-1].set(yticks=(min_y,), xlabel='Time (min)')
    # label y in middle axis
    axes[len(axes)//2].set(ylabel='Fluorescence (a.u.)')
    # adjust all x and y lim to be the same (y ax is not shared because of ticks)
    [ax.set(xlim=(min_x, max_x), ylim=(min_y, max_y)) for ax in axes]
    # remove all other ticks
    [ax.set(yticks=[]) for ax in axes[1:-1]]
    # bring plots closer
    plt.subplots_adjust(hspace=-hspace)
    sns.despine()

    if shade_peaks:
        shade_peaks_stackp(axes, shade_peaks, traces, t, **shade_peaks_kwargs)

    return fig, axes

def plot_peaks(peak_traces, t=20, alpha=0.2, color='k', mean_color='k', ax=None):
    """
    Plot individual peak traces in single plot
    """

    peak_time = np.arange(t, 1+peak_traces.shape[1]*t, t)/60
    mean_trace = peak_traces.mean(axis=0)
    if ax is None: fig, ax = plt.subplots()
    [ax.plot(peak_time, _p, '-', alpha=alpha, color=color) for _p in peak_traces]
    ax.plot(peak_time, mean_trace, '--', lw=5, alpha=0.8, color=mean_color)
    ax.set(xticks=np.arange(0, 7, 1), xlabel='Time (min)', ylabel='Fluorescence (a.u.)')
    sns.despine()
    plt.tight_layout()
    return ax

def tracking_movie(movie, tracks, x='x', y='y'):
    """
    Label particles being tracked on movie for visualization

    Arguments
    ---------
    movie: array
    tracks: pandas dataframe
        containing columns `x`, `y` and `frame` for each particle being tracked

    Returns
    ---------
    movie_tracks: array
        copy of movie with circles around each identified particle

    """

    def track_im(f, coords, im):
        coords = tracks[tracks.frame==f][[x, y]].dropna()
        im_plot = im.copy()
        try:
            circles = [circle_perimeter(int(c[y]), int(c[x]), 10,
                        shape=im.shape) for (_, c) in coords.iterrows()]
        # if nan, no coordinates specified for frame, just put image
        except ValueError:
            return im_plot
        for circle in circles:
            im_plot[circle] = np.max(im_plot)
        return im_plot

    movie_tracks = Parallel(n_jobs=6)(delayed(track_im)(f, tracks, im)
                           for f, im in tqdm(enumerate(movie)))

    return np.stack(movie_tracks)
