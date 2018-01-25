import pandas as pd
import numpy as np
import skimage
from skimage import io
from skimage.external.tifffile import TiffFile
import trackpy as tp
import pymc3 as pm
import bebi103
from bebi103_legacy import ecdf

from collections import defaultdict
from tqdm import tqdm
import os

from im_utils import *

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib


rrdir = '../data/FISH/20171201/'
ims_stack = load_ims(rrdir+'zstacks/', 'STK')
# load nuclei and cell markers
with open('../output/nuc_trainingset/20171201_nuclei_segment_centroids_markers_segim.pkl', 'rb') as f:
    seg_coords = pickle.load(f)
    sel_markers_dict = pickle.load(f)
# try it on a sample
#sample = ['666FISHGal10PP7_s10', '666FISHGal10PP7_s9']

# dataframe for single transcript peaks
peaks = pd.DataFrame()
for imname in tqdm(ims_stack):
#for imname in tqdm(sample):
    fish_stack = ims_stack[imname]
    # get quality controlled nuclei and cell markers
    cell_markers, nuclei_markers = sel_markers_dict[imname]

    # Identify peaks =========================================================
    # identify transcription particles, diameter of 3 works well
    # this params seem to work decently to identify single transcripts
    # Imaging params: LeicaImagingFacility, 100%int 300msExp 0.2uZstack
    # for 3D, below seems to work well
    print('identifying peaks...')
    _parts = tp.locate(fish_stack, 9, minmass=1000)
    _parts['imname'] = imname
    # Get total number of cells identified; 0 is background
    _parts['cell_number'] = len(np.unique(cell_markers)) - 1

    # Assign transcripts to cells ====================================================
    # Get cell label
    _parts['cell_label'] = _parts.apply(lambda coords:\
            cell_markers[int(coords.y), int(coords.x)], axis=1)
    # Get nuclear label
    _parts['nuc_label'] = _parts.apply(lambda coords:\
            nuclei_markers[int(coords.y), int(coords.x)], axis=1)

    peaks = pd.concat((peaks, _parts))
    print('done')

# filter peaks that are not in cells. If label>0, it is inside cell.
peaks = peaks[(peaks.cell_label>0)].reset_index(drop=True)
peaks['strain'] = peaks.imname.apply(lambda x: x.split('FISH')[0])
# assign unique cell and nuclear id
peaks['cid'] = peaks.apply(lambda x: x.imname+'_'+str(x.cell_label), axis=1)
peaks['nid'] = peaks.apply(lambda x: x.imname+'_'+str(x.nuc_label), axis=1)
peaks.to_csv('../output/{}_smFishPeaks3D.csv'.format(rrdir.split('/')[-2]), index=False)
