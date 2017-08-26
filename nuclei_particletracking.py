from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import skimage
from skimage import io
import os
from im_utils import *
import trackpy as tp
from tqdm import tqdm
from collections import defaultdict

# load z-projected movies
movs = {}
ddir = '../data/GFP_Envy_mScarlet/'
for fname in tqdm(os.listdir(ddir)):
    if 'tiff' in fname:
        movs[fname.split('.')[0]] = io.imread(ddir + fname)
    else: next

# area and intensity limits
maxint_lim, minor_ax_lim, major_ax_lim, area_lim = (0.2,1), (25, 500), (40, 500), (500, 5000)

# dictionary for time-projected mask, nuclei markers and properties
nuclei_proj = {}
# dataframe for transcription peaks and nuclei properties
nuclei_peaks = pd.DataFrame()

for mname in tqdm(movs):
    print('analyzing {}'.format(mname))
    movie = movs[mname][:5]
    print('normalizing...')
    movie_norm = normalize(movie) #np.stack([normalize(frame) for frame in movie])
    print('projecting through time axis...')
    movie_proj = z_project(movie_norm, 'max')
    print('creating binary mask...')
    # mask projected image using adaptive threshold to find nuclei of min_size
    m_mask = mask_image(movie_proj, min_size=200, block_size=101, selem=skimage.morphology.disk(15))
    print('filtering...')
    # get nuclei markers, bound by area and intensity
    markers_proj, _nuclei_proj = label_sizesel(movie_proj, m_mask,
                        maxint_lim, minor_ax_lim, major_ax_lim, area_lim)
    print('identifying peaks...')
    # identify transcription particles, diameter of 3 works well
    _parts = tp.batch(movie, 3)
    # enlarge nuclei markers to keep particles close to nuclear edge
    markers_proj_enlarged = skimage.morphology.dilation(markers_proj,
            selem=skimage.morphology.disk(5))
    # Get nuclear label
    _parts['label'] = _parts.apply(lambda coords:\
            markers_proj_enlarged[int(coords.y), int(coords.x)], axis=1)
    # remove peaks that are not in identified nuclei
    _parts = _parts[_parts.label>0]
    # track them
    print('tracking particles...')
    _parts = tp.link_df(_parts, 5, memory=1)
    # Dataframe for movie nuclei properties
    _nuclei = pd.DataFrame()
    print('measuring nuclei properties...')
    for frame_no, frame in enumerate(movie):
        # commented lines can be used to create a frame specific mask, which is
        # a more correct approach, but it doesn't really change much and adds
        # computation time (~3s per loop)
        ## create frame mask
        #mask_f = mask_image(frame, min_size=200, block_size=101, selem=skimage.morphology.disk(10))
        ## use time-projected, filtered markers to mark frame mask
        #markers_f = mask_f * markers_proj
        #nuclei_f = skimage.measure.regionprops(markers_f, frame)
        # get nuclei for each frame based on time-proj markers
        nuclei_f = skimage.measure.regionprops(markers_proj, frame)
        # convert to dataframe and save
        nuclei_fdf = regionprops2df(nuclei_f)
        nuclei_fdf['frame'] = frame_no
        _nuclei = pd.concat([_nuclei, nuclei_fdf])
    print('{0} total nuclei for {1}'.format(len(nuclei_f), mname))

    # merge, label and save
    _nuclei_peaks = pd.merge(_nuclei, _parts, on=['frame', 'label'])
    _nuclei_peaks['movie'] = mname
    # transform particle whole movie coordinates to nuclei bounding box coordinates
    _nuclei_peaks['x'] = _nuclei_peaks['x'] - _nuclei_peaks.bbox.apply(lambda x: x[1])
    _nuclei_peaks['y'] = _nuclei_peaks['y'] - _nuclei_peaks.bbox.apply(lambda x: x[1])
    nuclei_peaks = pd.concat([nuclei_peaks, _nuclei_peaks])
    # save markers and time-projected properties
    nuclei_proj[mname] = (_nuclei_proj, markers_proj)
    break

# get tracking movies for all nuclei

nuc_movs = defaultdict(list)
for name, nuc in tp.filter_stubs(nuclei_peaks, 5).groupby(['movie', 'label']):
    nuc_mov = np.stack(nuc.drop_duplicates('frame').intensity_image)
    nuc_trackmov = tracking_movie(nuc_mov, nuc)
    nuc_movs[name[0]].append(nuc_trackmov)
# make global nuclei tracking movie
globmov = concat_movies(nuc_movs['envy1'], nrows=3)
show_movie(globmov, 0.1, loop=True)

# make summary for less clutered, single summary plot
nsumm = nuclei.groupby(['frame', 'movie']).median().reset_index()
colors = iter(sns.color_palette("husl", 9))
fig, ax = plt.subplots(1)
for name, group in nsumm.groupby('movie'):
    group.plot(x='frame', y='mean_intensity', ax=ax, label=name, color=next(colors))

# plot actual nuclei of a given frame
fig, axes = plt.subplots(10, 10, sharex=True, sharey=True)
axesiter = iter(np.ravel(axes))
for n in nuclei['envy1'][-1]:
    ax = next(axesiter)
    ax.imshow(n.intensity_image, cmap='viridis')
    ax.set_xticks([])
    ax.set_yticks([])

# plot nuclei properties by movie and by frame
fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
axesiter = iter(np.ravel(axes))
for n, group in nuclei_int.groupby('movie'):
    ax = next(axesiter)
    group.groupby('label').plot(x='frame', y='median_int', ax=ax, alpha=0.1)
    # or swarmplot, but its pretty clutered
    #sns.swarmplot(x='frame', y='max_int', data=group, ax=ax, c='b', alpha=0.5)
    ax.legend('')
    ax.set_title(n)
plt.tight_layout()


# look at segmentation
fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
axes[0].imshow(movie_proj, cmap='viridis')
axes[1].imshow(markers_proj>0, cmap='viridis')
axes[2].imshow(movie_proj, cmap='viridis')
axes[2].imshow(markers_proj>0, alpha=0.1)

# try bandpass filters
fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
axes[0].imshow(envy1, cmap='viridis')
# 3 is reasonable (transcription) particle diameter for trackpy
# DO NOT threshold before gaussian bandpass; this will create artificial peaks!
#bpassed = filters.gaussian(envy1)
axes[1].imshow(tp.bandpass(envy1, 1, 5), cmap='viridis')
axes[2].imshow(tp.bandpass(envy1_m, 1, 3), cmap='viridis')


fig, axes = plt.subplots(2, sharex=True, sharey=True)
tp.annotate(parts, movie_proj, ax=axes[0])
tp.annotate(parts[parts.nucleus>0], movie_proj, ax=axes[1])

movie = movs['envy1']
test_traj = tp.link_df(parts_filtered[parts_filtered.movie=='envy1'], 5)
test_mov = tracking_movie(movie, test_traj[(test_traj.movie=='envy1')&(test_traj.particle==1.0)])
show_movie(test_mov, 0.01, loop=1)
io.imsave('test_nofilter.tiff', test_mov)

fig, axes = plt.subplots(3, sharex=True, sharey=True)
axes[0].imshow(markers_proj, cmap='viridis')
axes[1].imshow(markers_f, cmap='viridis')
axes[2].imshow(f, cmap='viridis')
