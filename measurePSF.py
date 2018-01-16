"""
Get width of point spread function (PSF) from mulitple z-stack images of 100 nm
fluorescent beads treated as single point light source. PSF width is defined
as the Full Width Half Maximum (FWHM) value of gaussian function fit.
Procedure:
    1. Identify each bead in image.
    2. Fit a 3D gaussian to each bead by minimizing the gaussian approximated
       negative log posterior to obtain optimal parameter values
       and error bars (standard deviation of posterior).
    3. Estimate most probable value of PSF width and
       error bars (standard deviation) from distribution of
       optimal values and errors.
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from im_utils import *
import scipy
from scipy.stats import multivariate_normal
import statsmodels.tools.numdiff as smnd
from matplotlib.mlab import griddata

# get (inverse) interpixel distance
meter = io.ImageCollection('../data/Leica_ImFacil/InterPixel/*um.tif')
interpixel_25um = interpixel_dist(meter[0], 25) # ~ 13.85 px/um or 72nm interpixel
interpixel_50um = interpixel_dist(meter[1], 50) # ~ 14.415 px/um
interpixel_100um = interpixel_dist(meter[1], 50) # ~ 14.416 px/um

beads = io.ImageCollection('../data/Leica_ImFacil/PSF/*.STK', load_func=TiffFile)
# use trackpy to find beads
beads = np.stack([b.asarray() for b in beads])
# This params work decent
#beads_pos = tp.locate(beads, 5, minmass=5000)
#beads_pos.to_csv('../data/Leica_ImFacil/PSF/beads_pos.csv', index=False)
beads_pos = pd.read_csv('../data/Leica_ImFacil/PSF/beads_pos.csv')
# take only those of 1 stdev around median to make it nice and normal
stdev, median = np.std(beads_pos.mass), np.median(beads_pos.mass)
beads_pos = beads_pos[(beads_pos.mass<median+stdev)&(beads_pos.mass>median-stdev)]
# Now this is normal distributed
plot_ecdf(beads_pos['mass'])

# Functions for optimization and checking fit =================================

def gauss1d(x, p):
    """ Unidimensional gaussian function """
    r, a, b, c = p
    ex = (x-b)**2/(2*c**2)
    return r + a * np.exp(-(ex))

def gauss3d(X, p):
    """ Three-dimensional gaussian function """
    z, y, x = X # Dimensions
    r, a, bx, by, bz, cx, cy, cz = p
    ex = (x-bx)**2/(2*cx**2)
    ey = (y-by)**2/(2*cy**2)
    ez = (z-bz)**2/(2*cz**2)
    return r + a * np.exp(-(ex + ey + ez))

def resid3d(p, X, y):
    """ Residuals for 3d Gaussian and measured values """
    return y - gauss3d(X, p)

def log_marg_like_gauss3d(p, X, y):
    """Log of marginalized likelihood"""
    return -len(X) / 2 * np.log(np.sum(resid3d(p, X, y)**2))

def log_marg_prior_gauss3d(p, p_range):
    """Log prior for 3D gaussian. Jeffreys on all c's, uniform on the rest"""

    # unpack parameters
    r, a, bx, by, bz, cx, cy, cz = p
    # put params and param ranges together
    p_prange = zip(p, p_range)

    def inbound(_param, _param_range):
        """ Check that `_param` is within bounds of `_param_range` """
        return _param_range[0] <= _param <= _param_range[1]

    if not all([inbound(_p, _p_range) for _p, _p_range in p_prange]):
        return -np.inf

    return -np.log(cx*cy*cz)

def log_marg_post_gauss3d(p, X, y, p_range):
    """ Log posterior for 3d gaussian """
    lp = log_marg_prior_gauss3d(p, p_range)

    if lp == -np.inf:
        return -np.inf

    return lp + log_marg_like_gauss3d(p, X, y)

def neg_log_post_gauss3d(p, X, y, p_range):
    """ Negative log posterior for 3d gaussian """
    return -log_marg_post_gauss3d(p, X, y, p_range)

# Regression by Optimization ==================================================

def imstack_gauss3d_regress(imstack, p0, p_range=None, return_err=True, leastsq=False):
    """
    Fit a 3D stack of images to a gaussian function by optimization

    Arguments
    ---------
    imstack: array
        Z-stack of images
    p0: array
        initial guesses for optimization
    p_range: array
        prior bounds on each parameter

    Returns
    -------
    popt: array
        result of optimization
    errbars: array
        error bars on optimal params

    """

    # Get indices of each pixel, i.e. z, y, x dimensions
    X = np.array([d.ravel() for d in np.indices(imstack.shape)])
    # Get fluorescence intensities for each (z, y, x) pixel
    fluor_int = imstack.ravel()

    # Least of squares regression on residuals
    if leastsq:
        popt3d, _ = scipy.optimize.leastsq(resid3d, p0, args=(X, fluor_int))
        return popt3d
    args = (X, fluor_int, p_range)
    # Compute the MAP
    popt = scipy.optimize.minimize(neg_log_post_gauss3d, p0,
                                        args=args, method='powell').x
    if return_err:
        hes = smnd.approx_hess(popt, log_marg_post_gauss3d, args=args)
        # Compute the covariance matrix and get error bars on params
        errbars = np.sqrt(np.diag(-np.linalg.inv(hes)))
        return popt, errbars
    else:
        return popt

# Params are: r, a, bx, by, bz, cx, cy, cz
# Initial guesses for regression. Center is roughly center of image.
p0 = (0.1, 1, *((wsize/2),)*3, 2, 2, 2)
# Parameter bounds, the same for all b's and c's
p_range = ((0,1), (0, 100), *((0, wsize),)*6)

# Dataframe to store optimized params and error bars by field of view
popt_all = pd.DataFrame()
# size of image window to get around predicted bead center
wsize = 11
#for frame_no, field in beads_pos.groupby('frame'):
# test on a sample
#beads_pos = beads_pos.sample(100)
# Get images of all beads
beads_ims_ = beads_pos.apply(lambda x: [get_bbox3d(x[['x','y','z']],
                    size=wsize, im=beads[x.frame.astype(int)])], axis=1)
# get shape of desired image window
imshape = ((wsize,))*3
# clear beads too close to 3D border, where couldn't retrieve full window
notinborder = beads_ims_.apply(lambda x: np.array_equal(x[0].shape, imshape))
beads_pos = beads_pos[notinborder]
beads_ims_ = beads_ims_[notinborder]
# scale them to [0,1] and put them in stack
beads_ims_ = np.stack([normalize_im(b[0]) for b in beads_ims_])
# regression by optimization on each bead
for i, _bead_im in enumerate(beads_ims_):
    # skip beads with misc errors in optimization (e.g. can't compute hessian)
    try:
        _popt, _errbars =  imstack_gauss3d_regress(_bead_im, p0, p_range)
    except:
        continue
    # Concatenate opt params, errbars and pos coordinates: x,y,z,field_of_view
    _popt_errbars_pos = np.concatenate((_popt, _errbars,
                        beads_pos.iloc[i][['frame','x','y','z']].astype(int)))
    # Add all to dataframe
    _popt_errbars_pos = pd.DataFrame(_popt_errbars_pos).T
    popt_all = pd.concat((popt_all, _popt_errbars_pos), ignore_index=True)

popt_all.columns = ('r', 'a', 'bx', 'by', 'bz','cx','cy','cz', 'r_err',
    'a_err', 'bx_err','by_err', 'bz_err', 'cx_err', 'cy_err', 'cz_err',
    'frame', 'x','y','z')

# get single estimate of c's by error propagation
# get variances and means in numpy arrays for convenience
c_vars = popt_all[['cx_err','cy_err','cz_err']].values**2
c_means = popt_all[['cx','cy','cz']].values
# propagate variance of c
c_var = 1/np.sum(1/c_vars, axis=0)
# get most prob value of c
c_mean = c_var * np.sum(c_means / c_vars, axis=0)
# get stdev of c for error bar
c_std = np.sqrt(c_var)
# compute Full Width Half Max (FWHM) with error bars, pixel width of PSF
FWHMxyz = 2*np.sqrt(2*np.log(2)) * np.concatenate((c_mean, c_std))
# convert to nanometers: 72nm interpixel distance, 200nm Z-step size
FWHMxyz_nm = np.multiply(FWHMxyz, (72,72,200,72,72,200))

# write it to file
PSFwidth_descrip =\
"""Leica DMI 6000 from Imaging Facility
100x objective
Interpixel distance: 72 nm
Point Spread Function (PSF) width for PENT 650-684 filter
(Full width half max) of error propagated gaussian fit from
{12} Tetraspeck fluorescent beads and 11 fields of view, in
each dimension:
    x = {0}+-{3} pixels = {6}+-{9} nm
    y = {1}+-{4} pixels = {7}+-{10} nm
    z = {2}+-{5} pixels = {8}+-{11} nm
    """.format(*FWHMxyz, *FWHMxyz_nm, popt_all.shape[0])
with open('../data/Leica_ImFacil/PSF/LeicaDMI600_650680_metadata.txt', 'w') as f:
    f.write(PSFwidth_descrip)

#popt_all.to_csv('../data/Leica_ImFacil/PSF/beads_3DGaussFit.csv', index=False)
popt_all = pd.read_csv('../data/Leica_ImFacil/PSF/beads_3DGaussFit.csv')

# Plot sigma over field of view; there should be single peak
popt_all['c_mean'] = popt_all.loc[:,['cx','cy']].mean(axis=1).values
fig, axes = plt.subplots(2, sharex=True, sharey=True)
# show sigma value with marker color
cm = plt.cm.viridis
cm_gradient = cm(np.linspace(0, 1, 10))
sns.regplot(x='x', y='y', data=popt_all.sort_values('c_mean'), fit_reg=0,
        scatter_kws={'alpha':0.5, 'color':cm_gradient ,'s':10}, ax=axes[0])

# Interpolate values and make filled contour plot
x, y, c_mean = popt_all[['x','y','c_mean']].values.T
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
zi = griddata(x, y, c_mean, xi, yi, interp='linear')
CS = axes[1].contourf(xi, yi, zi, cmap='viridis') #levels=np.linspace(zi.min(), zi.max(), 5),

#==============================================================================

#Until here PSF info, rest are (slower) alternative ways to do it or verify fit

#==============================================================================

# Check Fit ===============================================================

def checkGaussFit3d(bead_im, X, popt3d):
    """
    Make two plots to check 3d Gaussian fit in 2D and 1D max projection
    """
    # Get 3D image from optimal parameters
    fit3d = gauss3d(X, popt3d).reshape((-1, *bead_im.shape[:2]))
    # Plot projection of 3d fit
    fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
    axes[0,0].imshow(z_project(fit3d)) #xy
    axes[0,0].set_title('fit xy')
    axes[0,1].imshow(z_project(bead_im)) #xy
    axes[0,1].set_title('raw xy')
    axes[1,0].imshow(z_project(fit3d.T)) #yz
    axes[1,0].set_title('fit yz')
    axes[1,1].imshow(z_project(bead_im.T)) #xy
    axes[1,1].set_title('raw yz')

    # Now look at fit in one dimension, just sanity check
    # array for plotting
    xx = np.linspace(0, bead_im.shape[0], 1000)
    # unpack params
    r, a, bx, by, bz, cx, cy, cz = popt3d
    # values for 1D plot
    x = np.max(z_project(bead_im), axis=0)
    y = np.max(z_project(bead_im), axis=1)
    z = np.max(z_project(bead_im.T), axis=0)
    # plot fit for each dimension
    fig, axes = plt.subplots(3)
    dim = ('x','y','z')
    for i, (c_, b_, d) in enumerate(((cx, bx, x), (cy, by, y), (cz, bz, z))):
        axes[i].plot(xx, gauss1d(xx, (r, a, b_, c_)))
        axes[i].plot(np.linspace(0, len(d)-1, len(d)), d, '.')
        axes[i].set_title(dim[i])
        #axes[i].axvspan(b_-FWHMxyz[i]/2, b_+FWHMxyz[i]/2, facecolor='k', alpha=0.1)
    return None

checkGaussFit3d(bead_im, X, popt3d)

# Regression with MCMC ========================================================
# TOO SLOW

with pm.Model() as model:
    # Priors
    r = pm.Uniform('r', lower=0, upper=1)
    a = pm.Uniform('a', lower=0, upper=10)
    bx = pm.Uniform('bx', lower=0, upper=wsize)
    by = pm.Uniform('by', lower=0, upper=wsize)
    bz = pm.Uniform('bz', lower=0, upper=wsize)
    cx = bebi103.pm.Jeffreys('cx', lower=0.1, upper=wsize)
    cy = bebi103.pm.Jeffreys('cy', lower=0.1, upper=wsize)
    cz = bebi103.pm.Jeffreys('cz', lower=0.1, upper=wsize)
    p = (r, a, bx, by, bz, cx, cy, cz)
    sigma = bebi103.pm.Jeffreys('sigma', lower=0.1, upper=1)
    # Likelihood
    obs = pm.Normal('obs', mu=gauss3d(X, p), sd=sigma, observed=fluor_int)
    trace3d = pm.sample(draws=2000, tune=2000, njobs=4)

# Hierarchichal MCMC ========================================================
# NOT WORKING

X = np.array([d.ravel() for d in np.indices(beads_ims_[0].shape)])
fluor_int = np.stack([b.ravel() for b in beads_ims_])
ind = np.arange(len(beads_ims_))

# values for hyperpriors based on regression of a single bead
mu_r, tau_r = 0.07, 0.01
mu_a, tau_a = 1, 0.15
# same initial value (and sigma) for all dimensions, center of image
mu_b_xyz = wsize/2
tau_b_xyz = wsize/6
mu_cxy =  1.9
tau_cxy = mu_cxy/6
mu_cz = mu_cy - 0.2
tau_cz = mu_cz/6

tau_sigma_r = 0.03
tau_sigma_a = 1
tau_sigma_bxyz = wsize/2
tau_sigma_cxyz = 1.5

import theano.tensor as tt

def gauss3d(X, p):
    """ Three-dimensional gaussian function """
    z, y, x = X # Dimensions
    r, a, bx, by, bz, cx, cy, cz = p
    ex = tt.sqr(x-bx)/(2*tt.sqr(cx))
    ey = tt.sqr(y-by)/(2*tt.sqr(cy))
    ez = tt.sqr(z-bz)/(2*tt.sqr(cz))
    return r + a * tt.exp(-(ex + ey + ez))

with pm.Model() as hier_model:
    # Hyperpriors on mus
    r = pm.Normal('r', mu=mu_r, sd=tau_r, testval=mu_r)
    a = pm.Normal('a', mu=mu_a, sd=tau_a, testval=mu_a)

    bx = pm.Normal('bx', mu=mu_b_xyz, sd=tau_b_xyz, testval=mu_b_xyz)
    by = pm.Normal('by', mu=mu_b_xyz, sd=tau_b_xyz, testval=mu_b_xyz)
    bz = pm.Normal('bz', mu=mu_b_xyz, sd=tau_b_xyz, testval=mu_b_xyz)

    cx = pm.Normal('cx', mu=mu_cxy, sd=tau_cxy, testval=mu_cxy)
    cy = pm.Normal('cy', mu=mu_cxy, sd=tau_cxy, testval=mu_cxy)
    cz = pm.Normal('cz', mu=mu_cz, sd=tau_cz, testval=mu_cz)

    # Hyperpriors on sigmas
    sigma_r = pm.HalfNormal('sigma_r', tau_sigma_r, testval=tau_sigma_r)
    sigma_a = pm.HalfNormal('sigma_a', tau_sigma_a, testval=tau_sigma_a)

    sigma_bx = pm.HalfNormal('sigma_bx', tau_sigma_bxyz, testval=tau_sigma_bxyz)
    sigma_by = pm.HalfNormal('sigma_by', tau_sigma_bxyz, testval=tau_sigma_bxyz)
    sigma_bz = pm.HalfNormal('sigma_bz', tau_sigma_bxyz, testval=tau_sigma_bxyz)

    sigma_cx = pm.HalfNormal('sigma_cx', tau_sigma_cxyz, testval=tau_sigma_cxyz)
    sigma_cy = pm.HalfNormal('sigma_cy', tau_sigma_cxyz, testval=tau_sigma_cxyz)
    sigma_cz = pm.HalfNormal('sigma_cz', tau_sigma_cxyz, testval=tau_sigma_cxyz)

    # Parameter Priors, uncentered
    var_ri = pm.Normal('var_ri', mu=0, sd=1, shape=len(beads_ims_))
    ri = pm.Deterministic('ri', mu_r + var_ri*sigma_r)
    var_ai = pm.Normal('var_ai', mu=0, sd=1, shape=len(beads_ims_))
    ai = pm.Deterministic('ai', mu_a + var_ai*sigma_a)

    var_bxi = pm.Normal('var_bxi', mu=0, sd=1, shape=len(beads_ims_))
    bxi = pm.Deterministic('bxi', mu_bx + var_bxi*sigma_bx)
    var_byi = pm.Normal('var_byi', mu=0, sd=1, shape=len(beads_ims_))
    byi = pm.Deterministic('byi', mu_by + var_byi*sigma_by)
    var_bzi = pm.Normal('var_bzi', mu=0, sd=1, shape=len(beads_ims_))
    bzi = pm.Deterministic('bzi', mu_bz + var_bzi*sigma_bz)

    var_cxi = pm.Normal('var_cxi', mu=0, sd=1, shape=len(beads_ims_))
    cxi = pm.Deterministic('cxi', mu_cx + var_cxi*sigma_cx)
    var_cyi = pm.Normal('var_cyi', mu=0, sd=1, shape=len(beads_ims_))
    cyi = pm.Deterministic('cyi', mu_cy + var_cyi*sigma_cy)
    var_czi = pm.Normal('var_czi', mu=0, sd=1, shape=len(beads_ims_))
    czi = pm.Deterministic('czi', mu_cz + var_czi*sigma_cz)

    # Prior of sigma from likelihood
    sigma = pm.HalfNormal('sigma', 0.1, testval=0.1)
    # Expected
    #p = (ri[ind], ai[ind], bxi[ind], byi[ind], bzi[ind], cxi[ind], cyi[ind], czi[ind])
    p = (ri[0], ai[0], bxi[0], byi[0], bzi[0], cxi[0], cyi[0], czi[0])
    mu = gauss3d(X, p)

    # Likelihood
    obs = pm.Normal('obs', mu=mu, sd=sigma, observed=fluor_int)

with hier_model:
    trace3d_hier = pm.sample(draws=200, tune=200, njobs=4)


