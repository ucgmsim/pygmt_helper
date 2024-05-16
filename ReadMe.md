## pygmt helper package

This package contains a set of helper functions for [pygmt](https://www.pygmt.org/latest/) that allow for easy generation of NZ based spatial plots. Additionally, it also contains some functionality to generate some common plots, such as IM plots or fault traces plot.

The aim of this package is to make the common steps of creating a pygmt based spatial plot easy, for specific details the user can also add in some pygmt calls directly.

### Installation
Follow the installation instructions from [pygmt](https://www.pygmt.org/latest/install.html). 
Then install this package via pip.

### Run-Time
Running with NZ specific map data will result in much longer plotting times. Therefore, 
it is recommended to just it to None when iterating on the plot and only enable it for the final version.
Same applies for the high_res_topo option when loading the map data.

### General usage
```python
import pandas as pd

from pygmt_helper import plotting

# Load some spatial data
# with columns lat, lon, key
# where key is whatever you want to plot
data_df = pd.read_csv("bla")

# Create figure
fig = plotting.gen_region_fig("Title")

# Interpolate
grid = plotting.create_grid(data_df, "key")

# Plot the grid
plotting.plot_grid(fig, grid, "hot", (0, 10, (10 - 0) / 10))

# Add anything else to plot
# via pygmt
...

# Save the figure
fig.savefig(
    "/path_to_somwhere/figure.png",
    dpi=900,
    anti_alias=True,
)
```

### Examples
The example folder contains multiple complete (except of map data, which is part of qcore) examples plots.