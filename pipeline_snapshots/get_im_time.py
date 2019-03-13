import pandas as pd
from datetime import datetime
import glob
import re

part_dir = '../output/pipeline_snapshots/particles/parts_labeled.csv'
parts = pd.read_csv(part_dir)
# remove '_w2GFPlow' substring at the end of movie name
parts['mov_name'] = parts.mov_name.apply(lambda x: x[:-9])
processed_ims = parts.mov_name.unique()
# get image time
datadir = '/Volumes/PQdata/data/PP7/032019_PP7SnapShots/'
nd_paths = glob.glob(datadir+'*/*nd')
time_ims = []
for ndpath in nd_paths:
    im_name = ndpath.split('/')[-1][:-3]
    # filter out those not in parts
    if im_name not in processed_ims: continue
    with open(ndpath, 'r') as f:
        timeim = re.search(r'.*(\d\d:\d\d:\d\d)', f.read()).group(1)
    time_ims.append((im_name, timeim))
time_ims = pd.DataFrame(time_ims)
time_ims.columns = ['mov_name','time']
# get imaging session
time_ims['session'] = time_ims.mov_name.apply(lambda x: '_'.join(x.split('_')[:-1]))

# dictionary with initial time per imaging session
start_dict = time_ims.groupby('session').time.min().to_dict()
def get_time(t, session, start_dict=start_dict, t0=30, FMT='%H:%M:%S'):
    """ Get time deltas """
    tdelta = datetime.strptime(t, FMT) - datetime.strptime(start_dict[session], FMT)
    return tdelta.seconds/60 + t0

time_ims['time_postinduction'] = time_ims.apply(lambda x: get_time(x.time, x.session), axis=1)
# add time
# add time post induction
parts = pd.merge(parts, time_ims[['time_postinduction','mov_name']], on='mov_name')
parts.to_csv(part_dir, index=False)
