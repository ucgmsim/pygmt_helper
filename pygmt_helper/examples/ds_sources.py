from pathlib import Path

import pygmt
import pandas as pd
import numpy as np

from pygmt_helper import plotting


# Config
ds_ffp = Path(__file__).parent / "resources" / "NZ_DSmodel_2015.txt"
map_data_ffp = Path("path to qcore/qcore/data")
output_dir = Path("path to the output directory")


# If set to true then each rupture is weighted by recurance probability
incl_rec_prob = False

# Load the data
ds_df = pd.read_csv(ds_ffp)

split_ids = np.stack(
    np.char.split(ds_df.rupture_name.values.astype(str), "_", 2), axis=1
).T
ds_df["lat"] = split_ids[:, 0]
ds_df["lon"] = split_ids[:, 1]

# Plotting data
map_data = plotting.NZMapData.load(map_data_ffp) if map_data_ffp is not None else None

if incl_rec_prob:
    sum_df = (
        ds_df.groupby(["lat", "lon"])
        .sum()
        .reset_index()
        .loc[:, ["lat", "lon", "annual_rec_prob"]]
    )
    sum_df = sum_df.rename(columns={"annual_rec_prob": "sum"})

    cb_label = "Summed Reccurance Probability"
    cmap_limits = (0, 0.005, 0.005 / 10)
else:
    sum_df = (
        ds_df.groupby(["lat", "lon"]).count().reset_index().loc[:, ["lat", "lon", "mw"]]
    )
    sum_df = sum_df.rename(columns={"mw": "sum"})

    cb_label = "Number of ruptures"
    cmap_limits = (0, 120, 120 / 10)

fig = plotting.gen_region_fig(
    "Distributed Seismicity Sources",
    "NZ",
    map_data=map_data,
    plot_topo=True,
    plot_roads=False,
    plot_highways=True,
    plot_kwargs={"highway_pen_width": 0.1, "coastline_pen_width": 0.01},
)

pygmt.makecpt(
    cmap="hot",
    series=[cmap_limits[0], cmap_limits[1], cmap_limits[2]],
    reverse=True,
    log=False,
)
fig.plot(
    x=sum_df.lon.values.astype(float),
    y=sum_df.lat.values.astype(float),
    style="c0.07c",
    color=sum_df["sum"].values,
    cmap=True,
    transparency=10,
    pen="0.1p,black",
)

phase = f"+{cmap_limits[0]}" if cmap_limits[0] > 0 else f"+{cmap_limits[1]}"
cb_frame = [f"a+{cmap_limits[2] * 2}{phase}f+{cmap_limits[2]}"]
if cb_label is not None:
    cb_frame.append(f'x+l"{cb_label}"')
fig.colorbar(
    cmap=True,
    frame=cb_frame,
)

if incl_rec_prob:
    fig.savefig(
        output_dir / f"ds_sources_sum.png",
        dpi=900,
        anti_alias=True,
    )
else:
    fig.savefig(
        output_dir / f"ds_sources_count.png",
        dpi=900,
        anti_alias=True,
    )
