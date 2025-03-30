# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

sns.set_context("talk")

# Read the data
df = pd.read_csv("benchmark_summary.csv")

# Clean and prepare the data
df = df.dropna(subset=["Avg Time (s)"])
df["Is Fastest"] = df["Is Fastest"] == "True"
df["n_samples"] = df["n_samples"].astype(int)
df["n_features"] = df["n_features"].astype(int)
df.head()

# %%

# Create a figure with facets for each model type
fig, axes = plt.subplots(1, 3, figsize=(15, 9), sharey=True)

# Generate a palette with distinct colors for libraries
colors = {"scipy": "#1f77b4", "pyensmallen": "#ff7f0e", "statsmodels": "#2ca02c"}
linestyles = {5: "-", 20: "--"}
markers = {5: "o", 20: "^"}

# Get unique models
model_types = ["Linear", "Logistic", "Poisson"]

# Plot each model in a separate facet
for i, model_name in enumerate(model_types):
    ax = axes[i]
    model_data = df[df["Model"] == model_name]

    # Plot each feature count and library combination
    for n_features in [5, 20]:
        subset = model_data[model_data["n_features"] == n_features]

        for lib, color in colors.items():
            lib_data = subset[subset["Library"] == lib]
            if not lib_data.empty:
                # Sort by n_samples to ensure proper line connection
                lib_data = lib_data.sort_values("n_samples")

                ax.loglog(
                    lib_data["n_samples"],
                    lib_data["Avg Time (s)"],
                    marker=markers[n_features],
                    linestyle=linestyles[n_features],
                    color=color,
                    markersize=6,
                    linewidth=2,
                    label=f"{lib}, {n_features} features",
                )

    # Set titles and labels
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_title(f"{model_name} Regression")
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Set common x-axis label
axes[-1].set_xlabel("Number of Samples (log scale)")

# Set x-axis ticks
for ax in axes:
    ax.set_xticks([1000, 10000, 100000, 1000000, 10000000])
    ax.set_xticklabels(["1K", "10K", "100K", "1M", "10M"])

# Add explanatory annotations
for i, ax in enumerate(axes):
    model_name = model_types[i]
    min_y, max_y = ax.get_ylim()

    fastest_lib = None
    fastest_time = float("inf")

    # Find the fastest library for 10M samples
    for lib in colors.keys():
        for features in [5, 20]:
            lib_data = df[
                (df["Model"] == model_name)
                & (df["Library"] == lib)
                & (df["n_features"] == features)
                & (df["n_samples"] == 10000000)
            ]

            if not lib_data.empty and lib_data["Avg Time (s)"].iloc[0] < fastest_time:
                fastest_time = lib_data["Avg Time (s)"].iloc[0]
                fastest_lib = lib

# Add a main title to the figure
# fig.suptitle("Statistical Libraries Performance Comparison")
plt.tight_layout()
# plt.subplots_adjust(hspace=0.2)
#
# add legend outside of the plot
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0.0,
    title="Library, # of Features",
)


plt.savefig("library_performance_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
# %%
