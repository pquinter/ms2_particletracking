from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import bebi103

input_dir = '../output/{}_smFishPeaks3D.csv'.format('20171201')
output_dir = '../output/mrnaquant_20171201_succesfulRegressFloatFullZ.csv'

#==============================================================================
# Quality assurance of spots
#==============================================================================
# Classify with SVM
#==============================================================================
spot_df = pd.read_csv(input_dir)
# remove spots with low corr to ideal
spot_df = spot_df[spot_df.corrwideal>0.5].reset_index(drop=True)
# load classifier
with open('../output/spot_trainingset/clf_beta1_size9NormedNotSmooth.pkl', 'rb') as f:
    clf = pickle.load(f)

# load spots df
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

#=============================================================================
# Compute average mRNA intensity and discard outliers using good-bad data model
#=============================================================================
def goodbadMCMC(mrna_int, mu_, _mu, sigma_, _sigma, _sigma_bad):
    with pm.Model() as model:
        # Priors
        mu = pm.Uniform('mu', mu_, _mu)
        sigma = bebi103.pm.Jeffreys('sigma', sigma_, _sigma)
        sigma_bad = bebi103.pm.Jeffreys('sigma_bad', sigma, _sigma_bad)
        w = pm.Beta('w', alpha=0.5, beta=0.5, shape=len(mrna_int))
        # Likelihood is good-bad data model.
        a_obs = bebi103.pm.GoodBad('a_obs',
                                   mu=mu,
                                   sigma=sigma,
                                   sigma_bad=sigma_bad,
                                   w=w,
                                   observed=mrna_int)
        trace_goodbad = pm.sample(draws=2000, tune=2000, njobs=4)
    df_mcmc = bebi103.pm.trace_to_dataframe(trace_goodbad)
    # get prob of being bad (high w -> good, low w -> bad)
    wcols = [c for c in df_mcmc.columns if 'w_' in c]
    w = np.median(df_mcmc[wcols].values, axis=0)
    return df_mcmc, w
mrna_gaussint = spot_df.gauss_int
df_mcmc, w = goodbadMCMC(mrna_int, mu_=0, _mu=1, sigma_=0.01, _sigma=1,
        _sigma_bad=100)
mrna_massint = spot_df.mass
df_mcmc_mass, w_mass = goodbadMCMC(mrna_massint, mu_=0, _mu=50000, sigma_=500, _sigma=25000,
        _sigma_bad=50000)
# smaller valued intensities have higher prob of being bad
corner.corner(df_mcmc[['mu', 'sigma', 'sigma_bad']])
plt.figure()
plt.scatter(w, mrna_int, s=10, alpha=0.2)
spot_df['w_goodbad'] = w

# label based on good and bad model weight and correlation with ideal
spot_df['gb_label'] = ''
spot_df.loc[(spot_df.w_goodbad>0.5), 'gb_label'] = 'mrna'
single_mrna = df_mcmc.mu.median()
# crap: bad weight, low intensity
spot_df.loc[(spot_df.w_goodbad<0.5)&(spot_df.gauss_int<single_mrna), 'gb_label'] = 'crap'
# TS: bad weight, high intensity
spot_df.loc[(spot_df.w_goodbad<0.5)&(spot_df.gauss_int>single_mrna), 'gb_label'] = 'TS'

#========================================================================
# Manual curation
#========================================================================

spot_df['pid'] = spot_df.apply(lambda x: str(int(x.x))+str(int(x.y))+x.imname, axis=1)
mrna_qa = spot_df[(spot_df.svm_label.isin(['crap']))].sort_values(['corrwideal']).copy()
ind_mrna, ims_mrnaqa, mrna_qa_ = sel_training(mrna_qa, ims_proj, ncols=200,
        normall=1, mark_center=0, s=9, cmap='viridis', step=200)
# create new label for these in each round: crap_mrna, crap_mrna2...
spot_df.loc[spot_df.pid.isin(mrna_qa[ind_mrna].pid), 'svm_label'] = 'crap_TS'

# convert intensity to number of molecules
smrna_mass = spot_df[spot_df.gb_label=='mrna'].mass.median()
spot_df['no_mrnas_fit'] = 0
spot_df.loc[spot_df.gb_label=='mrna', 'no_mrnas_fit'] = 1
spot_df.loc[spot_df.gb_label=='TS', 'no_mrnas_fit'] = np.round(spot_df.gauss_int/single_mrna)
spot_df['no_mrnas_mass'] = np.round(spot_df.mass/smrna_mass)
spot_df.loc[spot_df.gb_label=='mrna', 'no_mrnas_mass'] = 1
fig, ax = plt.subplots(1)
plot_ecdf(spot_df.no_mrnas_mass, label='mass', ax=ax, formal=1, alpha=1)
plot_ecdf(spot_df.no_mrnas_fit, label='fit', ax=ax, formal=1, alpha=1)
plot_ecdf(spot_df[(spot_df.no_mrnas_fit>1)&(spot_df.svm_label=='TS')].no_mrnas_fit, label='fit', ax=ax, formal=1, alpha=1)
plt.legend()

spot_df.to_csv(output_dir, index=False)
