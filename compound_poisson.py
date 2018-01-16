from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as st
import pymc3 as pm
import bebi103
import corner

def compound_poisson(lfreq, lburst, samples=1000):
    """
    Simulate transcription as a compound poisson process
    """
    # bursts per cell; samples is number of cells
    burst_freq = st.poisson(lfreq).rvs(samples)
    # number of mRNA molecules of each burst (size)
    burst_size = [st.poisson(lburst).rvs(f) for f in burst_freq]
    # mRNA molecules per cell
    mrna_pcell = np.array([np.sum(s) for s in burst_size])
    return mrna_pcell

lfreq = np.linspace(0.5, 3, 4)
lburst = np.linspace(0.5, 3, 5)
lfreq = np.full_like(lburst, 2)
lambdas = zip(lfreq, lburst)
simul = [compound_poisson(l2, l1) for l1, l2 in lambdas]
fig, ax = plt.subplots(1)
[plot_ecdf(simul[i], ax, formal=1, alpha=1,label=lburst[i]) \
        for i in range(len(lburst))]
plt.legend()
#plt.hist(mrna_pcell, bins=100)
plot_ecdf(compound_poisson(0.1, 2))
plt.title('var freq')

def resid_cpoisson(p, y):
    lfreq, lburst = p
    return y - compound_poisson(lfreq, lburst)
simul = compound_poisson(5, 2)

with pm.Model() as model:
    # Priors
    lfreq = pm.Uniform('lfreq', lower=0, upper=100)
    lburst = pm.Uniform('lburst', lower=0, upper=10)
    # Likelihood
    burst_per_cell = pm.Poisson('burst_per_cell', mu=lfreq)
    mrna_per_burst = pm.Poisson('mrna_per_burst', mu=lburst, shape= )
    obs = pm.Poisson('obs', mu=lfreq, observed=simul)
    trace = pm.sample(draws=1000, tune=1000, njobs=4)
corner.corner(trace.get_values('lfreq'))

fig, ax = plt.subplots(1)
plot_ecdf(cpois_s, ax, label='5, 2')
n = df_mcmc.alpha.median()
p = df_mcmc.p.median()
plot_ecdf(nbion_s, ax, alpha=0.1)
plt.legend()
for i in range(100):
    cpois_s = compound_poisson(5, 2)
    nbinom_s = st.nbinom(n, p).rvs(1000)
    plt.scatter(np.sort(cpois_s), np.sort(nbinom_s), alpha=0.01)
x = np.arange(0, np.max(cpois_s+5))
plt.plot(x, x)


res=[]
for name, group in transcripts_bycell.groupby('strain'):
    with pm.Model() as nbinom_model:
        # Priors
        #alpha = pm.Uniform('alpha', lower=0, upper=100)
        #alpha = pm.HalfCauchy('alpha', 10)
        alpha = pm.Normal('alpha', mu=5, sd=1)
        mu = pm.HalfFlat('mu')
        # Likelihood
        obs = pm.NegativeBinomial('obs', mu=mu, alpha=alpha, observed=group.mrnas)
    with nbinom_model:
        trace = pm.sample(draws=2000, tune=2000, njobs=4)

    df_mcmc = bebi103.pm.trace_to_dataframe(trace, model=nbinom_model, log_post=True)
    # Compute p from samples
    df_mcmc['p'] = df_mcmc['alpha'] / (df_mcmc['alpha'] + df_mcmc['mu'])
    # Compute burst size from p
    df_mcmc['b'] = (1 - df_mcmc['p']) / df_mcmc['p']
    res.append((df_mcmc.alpha.median(), df_mcmc.b.median(), name))

corner.corner(df_mcmc[['alpha','b']])
