## MS2|PP7 Spot analysis

#### Data Extraction
1. Raw movies z-projection (`load_movies.py`)
2. Segment nuclei, identify and track spots(`nuclei_particletracking.py`).  This generates `nuclei_peaks` dataframe, saved as a `pickle`, which is then used for data analysis below.  


#### Exploratory Data Analysis
3. Generate tracking movie (`tracking_mov.py`)
4. Align transcriptional traces, generate trace heatmaps  (`tracking_mov.py`)
5. Plot nuclei fluorescence intensity properties  (`nucprops_plot.py`)

#### Sandbox
6. Tune segmentation parameters (`imsegmentation_checks.py`)
