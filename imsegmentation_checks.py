# used to refine params of nuclei_particletracking.py
# look at segmentation
fig, axes = plt.subplots(1,4, sharex=True, sharey=True)
axes[0].imshow(m_mask, cmap='viridis')
axes[1].imshow(movie_proj, cmap='viridis')
axes[2].imshow(markers_proj, cmap='tab20')
axes[3].imshow(movie_proj, cmap='viridis')
axes[3].imshow(markers_proj>0, alpha=0.1)

# try bandpass filters
envy1 = movs['gfp1'][0]
from scipy.ndimage.filters import uniform_filter1d
sigma, smoothing = 1, 3
boxcar = uniform_filter1d(envy1, 2*smoothing+1, axis=0,
                             mode='nearest', cval=0)
bpassed = tp.bandpass(envy1, 1, 3, 1)
fig, axes = plt.subplots(1,4, sharex=True, sharey=True, figsize=(22, 5))
axes[0].imshow(envy1, cmap='viridis')
# 3 is reasonable (transcription) particle diameter for trackpy
# DO NOT threshold before gaussian bandpass; this will create artificial peaks!
axes[1].imshow(boxcar, cmap='viridis')
axes[2].imshow(boxcar+bpassed, cmap='viridis')
axes[3].imshow(bpassed, cmap='viridis')

testc = (slice(1805, 1818, None), slice(1410, 1422, None))
testim = envy1[testc]
boxcar = uniform_filter1d(testim, 2*smoothing+1, axis=0,
                             mode='nearest', cval=0)
bpassed = tp.bandpass(testim, 1, 3, 1)

fig, axes = plt.subplots(1,4, sharex=True, sharey=True, figsize=(22, 5))
sns.heatmap(testim, cmap='viridis', ax=axes[0], annot=True, fmt='d')
sns.heatmap(boxcar, cmap='viridis', ax=axes[1], annot=True, fmt='d')
sns.heatmap(boxcar+bpassed, cmap='viridis', ax=axes[2], annot=True, fmt='d')
sns.heatmap(bpassed, cmap='viridis', ax=axes[3], annot=True, fmt='d')

plt.imshow(mask_cells+mask_nuclei.astype(int), cmap='viridis')
plt.imshow(cell_markers_enlarged, cmap='tab20')
plt.imshow(autof, alpha=0.8)
fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
axes[0,0].imshow(mask_cells, cmap='viridis')
axes[0,1].imshow(cell_markers_enlarged, cmap='viridis')
axes[1,0].imshow(nuclei_markers, cmap='tab20')
axes[1,1].imshow(dapi, cmap='viridis')
axes[1,1].imshow(mask_nuclei, alpha=0.1)
