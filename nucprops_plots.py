from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from bebi103_legacy import ecdf

nuclei_props = pd.read_pickle('../output/GFPEnvyScar/nuclei_peaks.p')
# plot nuclei properties by movie and by frame
colors = sns.color_palette('Blues')[::-1][:3]+sns.color_palette('BuGn_r')[:3]+sns.color_palette('magma')[:3]
fig = plt.figure(figsize=(14, 9))
nuclei_props['Time (s)'] = nuclei_props['frame']*20
sns.tsplot(time='Time (s)', value='mean_intensity', condition='movie',
        data=nuclei_props, estimator=np.nanmean, ci=95, color=colors)
plt.xlim(0, nuclei_props['Time (s)'].max())
plt.ylabel('Mean nuclear intensity (a.u.)')
sns.despine()
plt.tight_layout ()
#plt.savefig('../output/tsplot_GFPEnvyScar.pdf', bbox_inches='tight')

# plot whole movie ecdfs
fig = plt.figure(figsize=(10, 6))
i=0
for i, (mname, data) in enumerate(nuclei_props.groupby('movie')):
    plt.plot(*ecdf(data.mean_intensity), label=mname, c=colors[i])
plt.legend()
sns.despine()
plt.ylabel('ECDF')
plt.xlabel('Mean nuclear intensity (a.u.)')
plt.tight_layout()
#plt.savefig('../output/ecdf_GFPEnvyScar.pdf', bbox_inches='tight')


