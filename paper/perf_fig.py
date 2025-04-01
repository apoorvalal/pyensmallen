# %%
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

sns.set_context("talk")

######################################################################
# %%
with open("benchmark_results_20250331_215345.json", "r") as f:
    results = json.load(f)

# %%
def model_family_results(model_fam, size_key):
    m = results[model_fam][size_key]
    n_samples, n_features = map(int, size_key.replace('n', '').replace('k', '').split('_'))
    res = []
    i = 0
    for trial in m:
        i += 1
        for lib in trial:
            if lib != 'true':
                params = np.array(trial[lib]['params'])
                time, converged = trial[lib]['time'], np.isnan(params).sum() == 0
                res.append(
                    {
                        "model": model_fam,
                        "n": n_samples,
                        "k": n_features,
                        "trial_no": i,
                        "lib": lib,
                        "time": time,
                        "converged": converged,
                    }
                )
    return pd.DataFrame(res)
# %%
# Get the results for a specific model family and size
models = ["linear", "logistic", "poisson"]
keys = results['linear'].keys()
all_res = pd.concat(
    [model_family_results(model, key) for model in models for key in keys],
    axis=0,
)

# %%
agg_df = all_res.groupby(["model", "lib", "n", "k"]).agg(
    time_mean=("time", "mean"),
    converged_mean=("converged", "mean"),
).reset_index()

# %%
agg_df.n.unique()

# %%
# Create a combined column for lib (hue) and k (style)
agg_df['k'] = agg_df['k'].astype(str)  # Ensure k is a string for style differentiation
# %%

# Plot 1: Time mean faceted by model
g1 = sns.FacetGrid(agg_df.assign(z = ""), col='model', row = "k", height=6, aspect=1.2)
g1.map_dataframe(
    sns.lineplot,
    x='n',
    y='time_mean',
    hue='lib',
    style='z',
    markers=True,
    dashes=True,
)
# add grid lines
g1.map(plt.grid, linestyle='--', alpha=0.5)
g1.add_legend(title='Library, # of Features')
g1.set_axis_labels('Sample Size (log scale)', 'Runtime (seconds, log scale)')
g1.set_titles('Model: {col_name} | # of Features: {row_name}')

# Set y and x axis to log scale
g1.set(yscale='log')
g1.set(xscale='log')
g1.savefig('benchmark_time.png', dpi=300)
# %%
# Plot 1: Time mean faceted by model
g2 = sns.FacetGrid(agg_df.assign(z = ""), col='model', row = "k", height=6, aspect=1.2)
g2.map_dataframe(
    sns.lineplot,
    x='n',
    y='converged_mean',
    hue='lib',
    style='z',
    markers=True,
    dashes=True,
)
# add grid lines
g2.map(plt.grid, linestyle='--', alpha=0.5)
g2.add_legend(title='Library, # of Features')
g2.set_axis_labels('Number of Samples (log scale)', 'prop converged')
g2.set_titles('Model: {col_name} | # of Features: {row_name}')

# Set y and x axis to log scale
g2.set(xscale='log')
g2.savefig('benchmark_conv.png', dpi=300)
# %%



