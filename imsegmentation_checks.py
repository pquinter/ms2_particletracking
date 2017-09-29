# used to refine params of nuclei_particletracking.py
# look at segmentation
fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
axes[0].imshow(movie_proj, cmap='viridis')
axes[1].imshow(markers_proj>0, cmap='viridis')
axes[2].imshow(movie_proj, cmap='viridis')
axes[2].imshow(markers_proj>0, alpha=0.1)

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
