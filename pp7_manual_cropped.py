from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from utils import *
import pickle
import scipy

#mov= io.imread('/Users/porfirio/Desktop/PP7movies_AugSept2018/TL47pQC100vsyQC21/09132018/MAX_09132018_TL47pQC100vpQC21_150u10%int480&100u10%int560_GalT0_w1Brightfield_t1.TIF - GFPlow:560low.tif')
#roi_mov, rois = manual_roi_sel(mov)
output_dir = '../output/manual_cell_rois/09132018_TL47pQC100vyQC21.p'
# load selected rois
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
                wlen=25, prominence=s.std()) for s in traces]
# count peaks per cell
peaks_percell = np.array([len(p[0]) for p in peaks])
# get fluoresence trace of each peak
peak_traces = get_peak_traces(traces, peaks, win=10)
# get peak values
peak_vals = np.array([tr[p] for ps, tr in zip(peaks, trace) for p in ps[0]])

# plot everything
stackplot(traces, shade_peaks=peaks)
kymograph(roi_mov, vmin=100, vmax=None, remove_zeros=True)
plt.figure()
sns.heatmap(traces.T)
plot_peaks(peak_traces)

plot_ecdf(peaks_percell)
plot_ecdf(peak_vals)
