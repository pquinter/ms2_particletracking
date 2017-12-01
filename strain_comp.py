from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

# load dataframes
ddir = '../output/20171122/'
spot_df = pd.DataFrame()
for strain in os.listdir(ddir):
    spot_df = pd.concat((spot_df, pd.read_csv(ddir+strain)))
# put strain name in new column
spot_df['strain'] = spot_df.imname.apply(lambda x: x.split('FISH')[0])

# load classifier
with open('../output/spot_trainingset/clf_beta0.pkl', 'rb') as f:
    clf = pickle.load(f)

# Get spot images
rdir = '../data/FISH/20171122/'
ims_smooth = {}
for _dir in tqdm(os.listdir(rdir)):
    if 'DS_Store' in _dir: continue
    ddir = rdir + _dir + '/'
    for fname in tqdm(os.listdir(ddir)):
        if 'tif' in fname:
            _fname = fname.split('_')
            fish_im = io.imread(ddir + fname)[:,:,0]
            # Convert image to float
            fish_im = normalize_im(fish_im)
            # substract background with strong gaussian blur
            im_bg = skimage.filters.gaussian(fish_im, sigma=50)
            fish_im -= im_bg
            # smooth with gaussian
            fish_im = skimage.filters.gaussian(fish_im, sigma=1)
            ims_smooth[_fname[1]+'_'+_fname[-1].split('.')[0]] = fish_im
        else: next

# clear peaks near border
size=10
spot_df = spot_df[spot_df.apply(lambda x: check_borders(x[['x','y']],
                                        ims_smooth[x.imname], size), axis=1)]
# get spot images
spot_ims = spot_df.apply(lambda x: [get_bbox(x[['x','y']],
                ims_smooth[x.imname], size, pad=False)], axis=1)
# convert to stack and ravel for classification
spot_ims  = np.stack([i[0] for i in spot_ims])
spot_ims = np.stack([np.ravel(i) for i in spot_ims])
# classify
labels_pred = clf.predict(spot_ims)
# add labels
spot_df['svm_label'] = labels_pred
# discard bad samples (seems like didn't wash so well, in edge of coverslip)
spot_df = spot_df[~spot_df.imname.isin(('67f1FISHGal10PP7_s13', '664FISHGal10PP7_s10'))]


################################################################################

# count transcripts by cell
spot_df['cellid'] = spot_df['imname'] + spot_df.cell_label.apply(str)
transcripts_bycell = spot_df[spot_df.svm_label=='mrna'].groupby('cellid').x.count().reset_index()
transcripts_bycell['strain']  = transcripts_bycell.cellid.apply(lambda x: x.split('FISH')[0])
c = iter(sns.color_palette("Set2", 10))
for name, group in transcripts_bycell.groupby('strain'):
    _median, _mean = np.median(group.x), np.mean(group.x)
    label = '{0} median: {1:.2f} mean: {2:.2f}'.format(name, _median, _mean)
    plt.plot(*ecdf(group.x, conventional=True), c=next(c), label=label)
    ax.set_title(label)
plt.legend()

# TS int
for name, group in spot_df[(spot_df.svm_label=='TS')&(spot_df.mass>900)].groupby('strain'):
for name, group in TS_certain.groupby('strain'):
    _median, _mean = np.median(group.mass), np.mean(group.mass)
    plot_ecdf(group.mass, label='{0} median: {1:.2f} mean: {2:.2f}'.format(name, _median, _mean), formal=1)
plt.legend()

# Distribution of mrna int by sample
for name, group in spot_df[spot_df.svm_label=='TS'].groupby('imname'):
    if '68' in name: continue
    _median, _mean = np.median(group.mass), np.mean(group.mass)
    label = '{0} median: {1:.2f} mean: {2:.2f}'.format(name, _median, _mean)
    print(label)
    plot_ecdf(group.mass, label=label, formal=1)
plt.legend()



plot_ecdf(spot_df[spot_df.svm_label=='TS'].nuc_label.values)
plot_ecdf(spot_df[spot_df.svm_label=='TS'].mass.values)
plt.scatter(spot_df[spot_df.svm_label=='TS'].nuc_label.values>0,spot_df[spot_df.svm_label=='TS'].mass)

s=100
TS_certain = spot_df.sort_values('signal', ascending=False).groupby('cellid').apply(lambda x: x.iloc[0])
TS_certain = TS_certain[TS_certain.apply(lambda x: check_borders(x[['x','y']],
                                            ims_smooth[x.imname], s), axis=1)]
ts_certain_im = TS_certain.apply(lambda x: [get_bbox(x[['x','y']],
                            ims_smooth[x.imname], s)], axis=1)
ts_certain_im   = np.stack([i[0] for i in ts_certain_im ])
plt.figure()
io.imshow(im_block(ts_certain_im, 50, norm=0), cmap='viridis')

# look at SVM labels
_ims = spot_df[spot_df.svm_label=='TS'].apply(lambda x: [get_bbox(x[['x','y']],
                            ims_smooth[x.imname], 10)], axis=1)
_ims   = np.stack([i[0] for i in _ims])
plt.figure()
io.imshow(im_block(_ims, 30, norm=1), cmap='gist_stern')
