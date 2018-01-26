from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
# what distribution of correlations with ideal spot to expect?
idealspot = np.full((9,9), 0)
idealspot[4,4] = 1
idealspot = skimage.filters.gaussian(idealspot, sigma=4.2) # PSF width blur
noisescale = np.linspace(0, 1, 10)
fig, ax = plt.subplots(1)
for s in noisescale:
    noisyspots = []
    for i in range(10000):
        noise = np.random.normal(np.mean(idealspot),s*np.std(idealspot),(9,9))
        noisyspots.append(idealspot+noise)
    corrs = np.array([np.corrcoef(idealspot.ravel(), im.ravel())[1][0] for im in noisyspots])
    plot_ecdf(corrs, label=s, ax=ax)

noisyspots = np.stack(noisyspots)

