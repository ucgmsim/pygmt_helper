"""Generates an example ratio plot, with fake data"""

from pathlib import Path

import numpy as np
import pandas as pd

from pygmt_helper import plotting

site_ffp = Path(__file__).parent / "resources"

# Config
site_locations = (
    Path(__file__).parent
    / "resources"
    / "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll"
)
# output_dir = Path("path to the output directory")
output_dir = Path("./")

# Load locations
df = pd.read_csv(
    site_locations,
    sep=r"\s+",
    index_col=2,
    header=None,
    names=["lon", "lat"],
)

# Add some fake data
df["ratio"] = np.random.uniform(-0.5, 0.5, df.shape[0])

# Create the figure
fig = plotting.gen_region_fig("Ratio")

# Compute the interpolated grid
grid = plotting.create_grid(df, "ratio")

# Plot the grid
plotting.plot_grid(
    fig, grid, "polar", (-0.5, 0.5, 1.0 / 16), ("darkred", "darkblue"), transparency=35
)

# Save the figure
fig.savefig(
    output_dir / f"ratio.png",
    dpi=900,
    anti_alias=True,
)
