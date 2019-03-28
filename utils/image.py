import pandas as pd
import numpy as np
import scipy
import re

import skimage
from skimage import io
import skimage.filters
import skimage.segmentation
import scipy.ndimage
from skimage.external.tifffile import TiffFile

import trackpy as tp
from tqdm import tqdm
import datetime
import pickle
import os
import warnings
import glob

from joblib import Parallel, delayed
import multiprocessing

import im_utils

###############################################################################
# Pipeline
###############################################################################

def proj_mov(mov_dir, savedir):
    mov_name = re.search(r'.+/(.+)(?:\.tif)$', mov_dir).group(1)
    saveto = '{0}{1}.tif'.format(savedir, mov_name)
    saveto_ref = '{0}{1}_ref.tif'.format(savedir, mov_name)
    # check if projected movie already exists
    if os.path.isfile(saveto):
        warnings.warn('{} projection exists, skipping.'.format(mov_name))
        return None
    mov = io.imread(mov_dir)
    mov_proj = skimage.filters.median(im_utils.z_project(mov))
    mov_proj = mov_proj.copy()
    mov_proj = im_utils.remove_cs(mov_proj, perc=0.001, tol=2)
    io.imsave(saveto, mov_proj)
    io.imsave(saveto_ref, mov[10]) # as a reference if needed

def segment_image(im_path, savedir=None, maxint_lim=(100,500),
        minor_ax_lim = (15,500), major_ax_lim=(20,500), area_lim=(1000,5000)):
    im_name = re.search(r'.+/(.+)(?:\.tif)$', im_path).group(1)
    # check if segmented movie already exists
    if savedir is not None:
        seg_im_path = savedir+im_name+'.tif'
        if os.path.isfile(seg_im_path):
            warnings.warn('{} segmentation data exists, skipping.'.format(im_name))
            return None
    im = io.imread(im_path)
    # mask projected image using adaptive threshold to find nuclei of min_size
    m_mask = im_utils.mask_image(im, min_size=100, block_size=101,
                selem=skimage.morphology.disk(5), clear_border=True)
    # get nuclei markers with watershed transform, bound by area and intensity
    markers_proj, reg_props = im_utils.label_sizesel(im, m_mask,
        maxint_lim, minor_ax_lim, major_ax_lim, area_lim, watershed=True)
    # convert nuclei region properties to dataframe
    reg_props = im_utils.regionprops2df(reg_props, props=('label','centroid', 'area'))
    # expand centroid coordinates into x,y cols
    reg_props[['y_cell','x_cell']] = reg_props.centroid.apply(pd.Series)
    reg_props = reg_props.drop('centroid', axis=1)
    # enlarge nuclei markers to keep particles close to nuclear edge
    markers_proj = skimage.morphology.dilation(markers_proj,
            selem=skimage.morphology.disk(5))
    if savedir:
        io.imsave(seg_im_path, markers_proj)
        reg_props.to_csv(savedir+im_name+'.csv', index=False)
    return markers_proj, reg_props

def markers2rois(markers):
    """ Make ROI slice objects from markers """
    # get bounding boxes
    rois = [r.bbox for r in skimage.measure.regionprops(markers, coordinates='xy')]
    # convert to slice objects
    rois = [(slice(xy[0],xy[2]), slice(xy[1],xy[3])) for xy in rois]
    return rois

def load_zproject_STKimcollection(load_pattern, savedir=None, n_jobs=6):
    """
    Load collection or single STK files and do maximum intensity projection

    Arguments
    ---------
    load_pattern: str
        pattern of file paths
    savedir: str
        directory to save projected images

    Returns
    ---------
    projected: nd array or np stack
        projected images

    """
    collection = io.ImageCollection(load_pattern, load_func=TiffFile)
    # get names of any images already processed
    strain_dir = re.search(r'(\d+_(:?yQC|TL).+?)_', collection[0].filename).group(1)
    strain_dir = savedir + strain_dir + '/'
    proj_extant = [im.split('/')[-1][:-4] for im in glob.glob(strain_dir+'*tif')]
    # filter out those
    [warnings.warn('{} projection exists, skipping.'.format(p)) for p in proj_extant]
    collection = [im for im in collection if im.filename[:-4] not in proj_extant]
    zproj = lambda imname, im: (imname, im_utils.z_project(im))
    projected = Parallel(n_jobs=n_jobs)(delayed(zproj)
            (im.filename, im.asarray()) for im in tqdm(collection))
    if savedir:
        try: os.mkdir(strain_dir)
        except FileExistsError: pass
        [io.imsave(strain_dir+name[:-3]+'tif', im) for name, im in projected]
    return projected
##############################################################################
# Manual cell selection and segmentation
##############################################################################

def drawROIedge(roi, im, lw=2, fill_val='max'):
    """
    Draw edges around Region of Interest in image

    Arguments
    ---------
    roi: tuple of slice objects, as obtained from zoom2roi
    im: image that contains roi
    lw: int
        edge thickness to draw
    fill_val: int
        intensity value for edge to draw

    Returns
    --------
    _im: copy of image with edge around roi

    """
    _im = im.copy()
    # check if multiple rois
    if not isinstance(roi[0], slice):
        for r in roi:
            _im = drawROIedge(r, _im)
        return _im
    # get value to use for edge
    if fill_val=='max': fill_val = np.max(_im)
    # get start and end of rectangle
    x_start, x_end = roi[0].start, roi[0].stop
    y_start, y_end = roi[1].start, roi[1].stop
    # draw it
    _im[x_start:x_start+lw, y_start:y_end] = fill_val
    _im[x_end:x_end+lw, y_start:y_end] = fill_val
    _im[x_start:x_end, y_start:y_start+lw] = fill_val
    _im[x_start:x_end, y_end:y_end+lw] = fill_val
    return _im

def manual_roi_sel(mov, rois=None, cmap='viridis', plot_lim=5):
    """
    Manuallly crop multiple regions of interest (ROI)
    from max int projection of movie

    Arguments
    ---------
    mov: array like
        movie to select ROIs from
    rois: list, optional
        list of coordinates, to be updated with new selection

    Returns
    ---------
    roi_movs: array_like
        list of ROI movies
    rois: list
        updated list of ROI coordinates, can be used as frame[coords]

    """
    # max projection of movie to single frame
    mov_proj = skimage.filters.median(im_utils.z_project(mov))
    mov_proj = mov_proj.copy()
    mov_proj = im_utils.remove_cs(mov_proj, perc=0.001, tol=2)
    if rois: mov_proj = drawROIedge(rois, mov_proj)
    else: rois = []
    fig, ax = plt.subplots(1)
    ax.set(xticks=[], yticks=[])
    i=0
    while True:
        # make new figure to avoid excess memory consumption
        if i>plot_lim:
            plt.close('all')
            fig, ax = plt.subplots(1)
            ax.set(xticks=[], yticks=[])
        plt.tight_layout() # this fixes erratic drawing of ROI contour
        ax.set_title('zoom into roi, press Enter, then click again to add\nPress Enter to finish',
                fontdict={'fontsize':12})
        ax.imshow(mov_proj, cmap=cmap)
        _ = plt.ginput(10000, timeout=0, show_clicks=True)
        roi = zoom2roi(ax)
        # check if roi was selected, or is just the full image
        if roi[0].start == 0 and roi[1].stop == mov_proj.shape[0]-1:
            plt.close()
            break
        else:
            # mark selected roi and save
            mov_proj = drawROIedge(roi, mov_proj)
            rois.append(roi)
            # zoom out again
            plt.xlim(0, mov_proj.shape[1]-1)
            plt.ylim(mov_proj.shape[0]-1,0)
        i+=1
    roi_movs = [np.stack([f[r] for f in mov]) for r in rois]
    return roi_movs, rois

def refine_markers(rois, mov, n_jobs=multiprocessing.cpu_count()):
    """ Make movie mask with segmented regions only in ROIs """

    def mask_frame(f_number, rois, mov):
        """ Create intensity thresholded, limited to ROIs,
        labeled mask for frame """
        # background is 0, start labeled ROIs at 1
        mask = np.zeros_like(mov[0])
        labels_l, nuc_fluor_l = [], []
        for mask_val, _roi in enumerate(rois, start=1):
            _roi_im = mov[f_number][_roi]
            # intensity based segmentation inside ROI
            _roi_mask = im_utils.mask_image(_roi_im, min_size=100, block_size=101,
                    selem=skimage.morphology.disk(5), clear_border=False)
            # get median nuclear intensity
            labels_l.append(mask_val)
            nuc_fluor_l.append(np.median(_roi_im[_roi_mask]))
            # dilate segmented cells to keep particles on the edge
            _roi_mask = skimage.morphology.dilation(_roi_mask,
                        selem=skimage.morphology.disk(5))
            # limit segmented cell to ROI
            mask[_roi] = mask_val * _roi_mask
        nuc_fluor_df = pd.DataFrame({'roi':labels_l, 'nuc_fluor':nuc_fluor_l})
        nuc_fluor_df['frame'] = f_number

        return (mask, nuc_fluor_df)

    mask_fluor_list = Parallel(n_jobs=n_jobs)(delayed(mask_frame)(f_number, rois, mov)
        for f_number in tqdm(range(len(mov)), desc='generating masks'))
    # unpack mask and nuclear intensity
    mask_mov = np.stack([mi[0] for mi in mask_fluor_list])
    nuc_fluor = pd.concat([mi[1] for mi in mask_fluor_list])

    return mask_mov, nuc_fluor
