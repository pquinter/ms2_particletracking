from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Quality assurance of spots
# Classify with SVM first, then manually curate
# load classifier
with open('../output/spot_trainingset/clf_beta1_size9NormedNotSmooth.pkl', 'rb') as f:
    clf = pickle.load(f)

# load spots df
spot_df = pd.read_csv('../output/{}_smFishPeaks3D.csv'.format('20171201'))
spot_df['strain'] = spot_df.imname.apply(lambda x: x.split('FISH')[0])
# Get images
rrdir = '../data/FISH/20171201/'
ims_proj = load_ims(rrdir+'zprojected/', 'tif', channel=0)
# clear peaks near border
size=9
spot_df_ = spot_df[spot_df.apply(lambda x: check_borders(x[['x','y']],
                    ims_proj[x.imname], size), axis=1)].reset_index(drop=True)
# get spot images
spot_ims = get_batch_bbox(spot_df, ims_proj, size=size)
# normalize and ravel for classification
spot_ims  = normalize_im(spot_ims)
spot_ims = np.stack([np.ravel(i) for i in spot_ims])
# classify
labels_pred = clf.predict(spot_ims)
# add labels
spot_df['svm_label'] = labels_pred

#========================================================================
# manual Quality Assurance
spot_df['pid'] = spot_df.apply(lambda x: str(int(x.x))+str(int(x.y))+x.imname, axis=1)
mrna_qa = spot_df[(spot_df.svm_label.isin(['crap']))].sort_values(['corrwideal']).copy()
ind_mrna, ims_mrnaqa, mrna_qa_ = sel_training(mrna_qa, ims_proj, ncols=200, normall=1, mark_center=0, s=9, cmap='viridis')
# create new label for these in each round: crap_mrna, crap_mrna2...
spot_df.loc[spot_df.pid.isin(mrna_qa[ind_mrna].pid), 'svm_label'] = 'crap_TS'

fig, ax = plt.subplots(1)
for l in spot_df.svm_label.unique():
    plot_ecdf(spot_df[spot_df.svm_label==l].corrwideal, label=l, ax=ax)
plt.legend()

# do it in steps
mrna_qa = spot_df[(spot_df.svm_label.isin(['crap']))].sort_values(['cx']).copy()
ind_r, ims_r, mrna_qa_ = [], [], pd.DataFrame()
step = 2000
for n in np.arange(step, len(mrna_qa)+step, step):
    _ind_mrna, _ims_mrnaqa, _mrna_qa = sel_training(mrna_qa[n-step:n],
        ims_proj, ncols=60, mark_center=0, s=9, cmap='viridis', normall=1)
    print(n-step, n)
    ind_r.append(_ind_mrna)
    ims_r.append(_ims_mrnaqa)
    mrna_qa_ = pd.concat((mrna_qa_, _mrna_qa), ignore_index=True)
ind_r = np.concatenate(ind_r)
# create new label for these in each round: crap_mrna, crap_mrna2...
spot_df.loc[spot_df.pid.isin(mrna_qa_[ind_r].pid), 'svm_label'] = 'former_crap'

# quantify mrnas in TS
smrna_int = spot_df[spot_df.svm_label=='mrna'].gauss_int.median()
smrna_mass = spot_df[spot_df.svm_label=='mrna'].mass.median()
spot_df['no_mrnas'] = np.round(spot_df.gauss_int/smrna_int)
spot_df['no_mrnas'] = np.round(spot_df.mass/smrna_mass)
spot_df.loc[(spot_df.no_mrnas<1), 'no_mrnas'] = 1
