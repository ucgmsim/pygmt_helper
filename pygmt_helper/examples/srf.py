"""
Example script that shows how to utilise the
pygmt_helper package for visualising of an SRF file
"""

from pathlib import Path

import pandas as pd
import numpy as np

from pygmt_helper import plotting
from qcore import srf


map_data_ffp = Path("/path/to/qcore/qcore/data")
output_dir = Path("/path/to/output_dir")

srf_ffp = Path(__file__).parent / "resources" / "Akatarawa.srf"

# Load the srf data
planes = srf.read_header(str(srf_ffp), idx=True)
corners = srf.get_bounds(str(srf_ffp), depth=True)
corners = np.transpose(np.array(corners), (1, 2, 0))
slip_values = srf.srf2llv_py(str(srf_ffp), value="slip")

# Set map_data to None for faster plotting without topography
map_data = plotting.NZMapData.load(map_data_ffp, high_res_topo=False)
# map_data = None

# Compute colormap limits
slip_all = np.concatenate([cur_values[:, 2] for cur_values in slip_values])
slip_cb_max = int(np.round(np.quantile(slip_all, 0.98), -1))
cmap_limits = (0, slip_cb_max, slip_cb_max / 10)

region = (
    corners[:, 0, :].min() - 0.5,
    corners[:, 0, :].max() + 0.5,
    corners[:, 1, :].min() - 0.25,
    corners[:, 1, :].max() + 0.25,
)
fig = plotting.gen_region_fig("Title", region=region, map_data=map_data)

# Process each fault plane
for ix, (cur_plane, cur_slip) in enumerate(zip(planes, slip_values)):
    print(f"Processing plane {ix}")
    cur_corners = corners[:, :, ix]

    # Turn into a grid
    cur_df = pd.DataFrame(data=cur_slip, columns=["lon", "lat", "slip"])
    cur_xmin, cur_xmax = cur_corners[:, 0].min(), cur_corners[:, 0].max()
    cur_ymin, cur_ymax = cur_corners[:, 1].min(), cur_corners[:, 1].max()
    cur_grid = plotting.create_grid(
        cur_df,
        "slip",
        grid_spacing="5e/5e",
        region=(cur_xmin, cur_xmax, cur_ymin, cur_ymax),
        set_water_to_nan=False,
    )

    # Plot the grid
    plotting.plot_grid(
        fig,
        cur_grid,
        "hot",
        cmap_limits,
        ("white", "black"),
        transparency=0,
        reverse_cmap=True,
        plot_contours=True,
        cb_label="Slip",
        continuous_cmap=True,
    )

    # Draw the outline
    fig.plot(
        x=cur_corners[:, 0].tolist() + [cur_corners[0, 0]],
        y=cur_corners[:, 1].tolist() + [cur_corners[0, 1]],
        pen="0.5p,black,-",
    )

    # Draw the top-edge
    cur_sort_ind = np.argsort(cur_corners[:, 2])
    fig.plot(
        x=cur_corners[cur_sort_ind, 0][:2],
        y=cur_corners[cur_sort_ind, 1][:2],
        pen="1.0p,black",
    )

fig.savefig(
    output_dir / f"srf.png",
    dpi=1200,
    anti_alias=True,
)
