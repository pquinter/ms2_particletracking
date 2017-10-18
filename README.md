## MS2|PP7 Spot analysis 

#### Data Extraction
1. Raw movies z-projection (`load_movies.py`).
2. Segment nuclei, identify and track spots(`nuclei_particletracking.py`).  This generates `nuclei_peaks` dataframe, saved as a `pickle`, which is then used for data analysis below.  


#### Exploratory Data Analysis
3. Generate tracking movie (`tracking_mov.py`).
4. Align transcriptional traces, generate trace heatmaps  (`trtrace_analysis.py`).
5. Plot nuclei fluorescence intensity properties  (`nucprops_plot.py`).

#### Sandbox
6. Tune segmentation parameters (`imsegmentation_checks.py`).

#### Requirements
[Anaconda](https://www.anaconda.com/download/#macos) distribution with `Python 3.6`.

Not included in default Anaconda installation:

[Trackpy](http://soft-matter.github.io/trackpy/v0.3.2/installation.html)
[tqdm](https://pypi.python.org/pypi/tqdm)
[im_utils](https://github.com/pquinter/PQpython_utils)
