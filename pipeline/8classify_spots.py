import pandas as pd
from utils import particle
import corner

part_dir = '../output/pipeline/particles'
parts = pd.read_csv(part_dir+'/parts_filtered.csv')
clf_scaler_path = '../output/pipeline/GPClassification/GPCclfRBF.p'
parts['GPCprob'] = particle.predict_prob(parts, ['corrwideal','mass_norm'], clf_scaler_path)
# get common particle pid (unique per particle but not per frame)
parts['cpid'] = parts['mov_name']+'_'+parts['particle'].apply(str)
parts.to_csv(part_dir+'/parts_labeled.csv', index=False)

plot_dir='../output/pipeline/GPClassification/plots/classif/'
# plot number of particles left vs prob. threshold
prob_thresh = np.linspace(0.1, 1, 100, endpoint=False)
parts_left = []
for thresh in prob_thresh:
    goodparts = parts[(parts.GPCprob>thresh)].cpid.values
    filt_ix = parts.cpid.isin(goodparts)
    parts_left.append(filt_ix.sum())

fig, ax = plt.subplots()
ax.scatter(prob_thresh, parts_left, alpha=0.3)
ax.set(xlabel='Prob. threshold', ylabel='No. particles left')
sns.despine()
plt.tight_layout()
plt.savefig(plot_dir+'partsleft_byprob.pdf', bbox_inches='tight')

# plot 2D distribution of fluor and correlation with ideal after 0.5 prob filter
goodparts = parts[(parts.GPCprob>0.5)].cpid.values
fig = corner.corner(parts[~filt_ix][['mass_norm','corrwideal']], color='r', alpha=0.8,
    labels=['Fluor/Nuc. Fluor', 'Correlation with ideal'], hist_kwargs={'density':True})
corner.corner(parts[filt_ix][['mass_norm','corrwideal']], color='b', fig=fig, hist_kwargs={'density':True}, alpha=0.1)
sns.despine()
plt.tight_layout()
plt.savefig(plot_dir+'corner_05ProbThresh.pdf', bbox_inches='tight')


