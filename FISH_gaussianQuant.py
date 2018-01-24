from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#========================================================================
# Fit spots to 3d Gaussian
#========================================================================
# get full 3d images
ims_stack = load_ims(rrdir+'zstacks/', 'STK')
# get spot 3D patches from images
spot_df = pd.read_csv('../output/{}_smFishPeaks3D.csv'.format('20171201'))
spot_ims = get_batch_bbox(spot_df, ims_stack, wsize, im3d=True, size_z='Full')
# convert to float for regression
spot_ims_float = [skimage.img_as_float(im) for im in spot_ims]

# Dataframe to store optimized params and error bars
popt_all = pd.DataFrame()
# Initial guesses for regression. Center is roughly center of image.
# Params are: r, a, bx, by, bz, cx, cy, cz
# r is offset, a is coeff, b are center coord, c are sigmas
p0 = (np.median(np.concatenate([i.ravel() for i in spot_ims_float])), 0.05, *((wsize/2),)*3, 2, 2, 2)
# Parameter bounds, the same for all b's and c's
p_range = ((0,1), (0, 1), *((0, wsize),)*6)
for i, ts_im in tqdm(enumerate(spot_ims_float)):
    try: #minimizing the negative log posterior
        _popt, _err =  imstack_gauss3d_regress(ts_im, p0, p_range)
    except: #least of squares is more robust
        try:
            _popt =  imstack_gauss3d_regress(ts_im, p0, leastsq=1)
        except: # if leastsq fails, fill with nan
            _popt = np.full_like(p0, np.nan)
    # Add to dataframe
    popt_all = pd.concat((popt_all, pd.DataFrame(_popt).T), ignore_index=True)
popt_all.columns = ('r', 'a', 'bx', 'by', 'bz','cx','cy','cz')
spot_df = pd.concat((spot_df, popt_all), axis=1)

# compute intensities and with subtracted linear background (r)
fit_ints = popt_all.apply(lambda x: gauss3d(X, x).sum() - x.r*wsize**3, axis=1)
spot_df['gauss_int'] = fit_ints

# compute correlation to an ideal spot: point source blurred with gaussian of PSF width
ims_proj_fish = load_ims(rrdir+'zprojected/', 'tif', channel=0)
allims = get_batch_bbox(spot_df, ims_proj_fish)
idealspot = np.full((9,9), 0) # zero filled wsize*wsize array
idealspot[4,4] = 1 # single light point source at the center
idealspot = skimage.filters.gaussian(idealspot, sigma=4.2) # PSF width blur
# pearson corr on projected im is best, assuming im is centered at potential
# peak. This is usually eq to max of normalized cross-correlation.
# Also tried spearman and 3d stacks, slower and not worth it.
allcorrs = np.array([np.corrcoef(idealspot.ravel(), im.ravel())[1][0] for im in allims])
spot_df['corrwideal'] = allcorrs
spot_df.to_csv('../output/mrnaquant_20171201_succesfulRegressFloatFullZ.csv', index=False)
