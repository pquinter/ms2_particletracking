from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from utils import *
import pickle
import scipy

#mov= io.imread('/Users/porfirio/Desktop/PP7movies_AugSept2018/TL47pQC100vsyQC21/09132018/MAX_09132018_TL47pQC100vpQC21_150u10%int480&100u10%int560_GalT0_w1Brightfield_t1.TIF - GFPlow:560low.tif')
mov = io.imread('/Users/porfirio/Desktop/PP7movies_AugSept2018/TL47pQC100vsyQC21/09122018/C1-MAX_09122018_TL47pQC100vpQC21_100u10%int480&560_1_w1Brightfield_t1.TIF - GFPlow:560low.tif')
mov = io.imread('../data/02122019_yQC23_100u10%int480_1.tif')
#mov = io.imread('../data/02122019_yQC22_100u10%int480_1.tif')
roi_mov, rois = manual_roi_sel(mov)
ods = ['../output/manual_cell_rois/09132018_TL47pQC100vyQC21.p',
 '../output/manual_cell_rois/09122018_TL47pQC100vyQC21_100u10%int.p',
 '../output/manual_cell_rois/02122019_yQC22_100u10%int480_1.p',
 '../output/manual_cell_rois/02122019_yQC23_100u10%int480_1.p']
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
c = iter(['k','b','r','g'])
#with open(output_dir, 'wb') as f:
#    pickle.dump(roi_mov, f)
#    pickle.dump(rois, f)
    # load selected rois
for output_dir in ods:
    with open(output_dir, 'rb') as f:
        roi_mov = pickle.load(f)
        rois = pickle.load(f)

    # get fluorescence traces and smooth
    traces = get_traces(roi_mov, plot=True, norm=True, pad=None,
                        smooth=True, smooth_win=3, smooth_func=np.mean)

    # find peaks
    # prominence ~0.5-1 std works well if normalized by mean
    # prominence abs value roughly 0.2 when normalized by mean; 30 when raw
    peaks = [scipy.signal.find_peaks(s, width=5, rel_height=1, distance=3,
                    wlen=15, prominence=1.5*s.std()) for s in traces]
    # count peaks per cell
    peaks_percell = np.array([len(p[0]) for p in peaks])
    # get fluoresence trace of each peak
    peak_traces = get_peak_traces(traces, peaks, win=10)
    # get peak values
    peak_vals = np.array([tr[p] for ps, tr in zip(peaks, traces) for p in ps[0]])

    color=next(c)
    plot_ecdf(peak_vals, color=color, ax=ax)
    plot_peaks(peak_traces, ax=ax2, color=color, mean_color=color)

    # plot everything
    stackplot(traces)
    kymograph(roi_mov, vmin=100, vmax=None, remove_zeros=True)
    plt.figure()
    sns.heatmap(traces.T)
    plt.figure()
    io.imshow(im_block(roi_mov[0], 40, norm=False), cmap='viridis')
    plot_ecdf(peaks_percell)

import trackpy as tp
# make mask with manually selected ROIs
mask = np.zeros_like(mov[0])
for i, _roi in enumerate(rois):
    mask[_roi] = i
# identify transcription particles
_parts = tp.batch(mov, 3, minmass=10, characterize=False, noise_size=1, smoothing_size=5)
# assign roi number to each spot
_parts['roi'] = _parts.apply(lambda coords: mask[int(coords.y), int(coords.x)], axis=1)
# remove spots outside ROIs
_parts = _parts[_parts.roi>0]
# only keep the n=2 brightest spots per ROI per frame
_parts_filt = _parts.sort_values('mass').groupby(['roi','frame']).tail(2).reset_index(drop=True)
_parts_track = tp.link_df(_parts_filt, 5, memory=1)
# compute trajectory length for each particle
track_len = _parts_track.groupby('particle').count().reset_index()[['particle','x']]
track_len.columns = ['particle','track_len']
_parts_track = pd.merge(_parts_track, track_len, on='particle')

# keep only ONE spot per frame: with longest track OR brightest, in that order
_parts_filt = _parts_track.sort_values(['track_len','mass']).drop_duplicates(['roi','frame'], keep='last')
# keep only spots that are at least 3 frames long
_parts_filt = _parts_filt[_parts_filt.track_len>3]
# make imputation dataframe with summary statistic of each roi
#TODO: part mass is much lower than mean value because bk subtraction, fix imputation value
_frames_n, _rois_n, _values = [], [], []
for _roi_number, _roi_mov in enumerate(roi_mov):
    for f_number, _roi_frame in enumerate(_roi_mov):
        _frames_n.append(f_number)
        _rois_n.append(_roi_number)
        _values.append(np.mean(mov[f_number]))
imput_df = pd.DataFrame()
imput_df['frame'] = _frames_n
imput_df['roi'] = _rois_n
imput_df['mass'] = 10
imput_df['imput'] = True
_parts_filt['imput'] = False
_parts_filt = pd.concat((_parts_filt, imput_df), sort=False)
# keep only ONE spot per frame, impute if none available
_parts_filt_ = _parts_filt.sort_values('imput').drop_duplicates(['roi','frame'], keep='first')

traces = _parts_filt_.pivot(index='roi', columns='frame', values='mass').values
traces_s = smooth_traces(traces, smooth_win=4)
plt.figure()
sns.heatmap(traces_s)
stackplot(traces_s)
stackplot(test)


_parts_track_filt = tp.filter_stubs(_parts_track, threshold=3)

plt.figure()
tp.annotate(_parts_filt[_parts_filt.frame==75], mov[75])
plt.figure()
tp.annotate(_parts_filt, proj)

track_mov = tracking_movie(mov, _parts_filt)
io.imsave('/Users/porfirio/Desktop/trackingmov.tif', track_mov)
