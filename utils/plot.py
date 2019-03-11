from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import glob

import skimage
from skimage import io
import im_utils
import scipy

from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import numba

from skimage.draw import circle_perimeter
import matplotlib.patches as mpatches
import corner

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

def tracking_movie(movie, tracks, n_jobs=multiprocessing.cpu_count(), x='x', y='y'):
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

    movie_tracks = Parallel(n_jobs=n_jobs)(delayed(track_im)(f, tracks, im)
                           for f, im in tqdm(enumerate(movie)))

    return np.stack(movie_tracks)

def plot2dDecisionFunc(clf, xs, ys, colors=('b','r'), labels=(True, False),
        xlabel='correlation with ideal spot', ylabel='intensity', 
        plot_data_contour=True, plot_scatter=True, scatter_alpha=0.1, save=False):
    """
    Plot decision surface of classifier with 2D points on top
    """
    # transpose data if necessary
    if len(xs)>len(xs.T): xs = xs.T

    # make grid
    xx, yy = np.meshgrid(
            np.linspace(np.min(xs[0]), np.max(xs[0]), 100),
            np.linspace(np.min(xs[1]), np.max(xs[1]), 100))
    # get decision function
    if hasattr(clf, 'decision_function'):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    z = z.reshape(xx.shape)

    # put things in dataframe
    data = pd.DataFrame()
    data[xlabel] = xs[0]
    data[ylabel] = xs[1]
    data['y'] = ys

    colors = dict(zip(labels, colors))
    # make the base figure for corner plot
    ndim = len(xs)
    fig, axes = plt.subplots(ndim, ndim, figsize=(12, 10))

    if not plot_data_contour: axes[1,0].clear()

    if hasattr(clf, 'decision_function'):
        # plot decision boundary
        axes[1,0].contour(xx, yy, z, levels=[0], linewidths=2, colors='#FFC300')
    else:
        # or probability distribution
        cs = axes[1,0].contourf(xx, yy, z, cmap='viridis')
    handles = [mpatches.Patch(color=colors[l], label=l) for l in labels]
    axes[1,1].legend(handles=handles)

    # plot data with corner
    data.groupby('y').apply(lambda x: corner.corner(x, color=colors[x.name], hist_kwargs={'density':True}, fig=fig))
    # plot data on top
    if plot_scatter:
        data.groupby('y').apply(lambda x: axes[1,0].scatter(x[xlabel],
                    x[ylabel], alpha=scatter_alpha, color=colors[x.name]))

    # add colorbar to countourf. Must be done after corner, or it will complain
    if hasattr(clf, 'predict_proba'):
        fig.colorbar(cs, ax=axes[1,0], ticks=np.linspace(0,1,5))
    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches='tight')

###############################################################################
# ECDF and boostrapping
###############################################################################
@numba.jit(nopython=True)
def draw_bs_sample(data):
    """
    Draw a bootstrap sample from a 1D data set.
    """
    return np.random.choice(data, size=len(data))

def bs_fromdf(df, groupby, col, no_bs):
    # Make 100 bootstrap samples
    bs_dict = {}
    for name, data in df.groupby(groupby):
        bs_samples = Parallel(n_jobs=12)(delayed(draw_bs_sample)
                (data[col].values) for _ in tqdm(range(no_bs)))
        bs_dict[name] = bs_samples
    return bs_dict

def bs_fromdict(data_dict, no_bs):
    # Make 100 bootstrap samples
    bs_dict = {}
    for name in data_dict:
        bs_samples = Parallel(n_jobs=12)(delayed(draw_bs_sample)
                (np.array(data_dict[name])) for _ in tqdm(range(no_bs)))
        bs_dict[name] = bs_samples
    return bs_dict


def ecdfs_fromdict(bs_dict):
    # turn into ecdfs
    ecdf_dict = {}
    for strain in bs_dict:
        ecdfs = Parallel(n_jobs=12)(delayed(im_utils.ecdf)(data)
                for data in tqdm(bs_dict[strain]))
        ecdf_dict[strain] = ecdfs
    return ecdf_dict

def plot_ecdfdict(ecdf_dict, ax=None, colors=None, alpha=0.05):
    if ax is None: fig, ax = plt.subplots()
    # and plot them
    for name in ecdf_dict:
        for ecdf in tqdm(ecdf_dict[name]):
            if colors is not None:
                try: color = colors[name]
                except TypeError: color=colors
            ax.plot(*ecdf, alpha=alpha, color=color)
    return ax
