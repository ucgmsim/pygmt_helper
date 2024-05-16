""""Script for plotting sites and historic events"""
from pathlib import Path

import numpy as np
import pandas as pd

from pygmt_helper import plotting

### Config
# Path to the events csv file, required columns are [lon, lat, mag]
# Easiest option is to use the earthquake_source_table.csv from the NZGMDB
# but any csv file with the required columns will work
events_ffp = Path("path to the events csv file")
# Path to the sites csv file, required columns are [lon, lat]
# Again, easiest option is to use the site_table.csv from the NZGMDB
# but any csv file with the required columns will work
sites_ffp = Path("path to the sites csv file")
# Region of interest to plot
region = [171.54, 173.12, -43.95, -43.22]
# Region of the inset
inset_region = [172.60, 172.69, -43.545, -43.495]
# Output file name
output_ffp = Path("path to the output file")
# Path to the map data
# Is part of the qcore package, once installed it
# can be found under qcore/qcore/data
# Set to None for lower quality map, but much faster plotting time
map_data_ffp = None
# If true, then use the high resolution topography
# This will further increase plot time, and only has an
# effect if map_data_ffp is set
use_high_res_topo = True
# List of events to highlight
events_to_highlight = None
# List of sites to highlight
sites_to_highlight = None
# Minimum magnitude to plot
min_mag = 4
### End Config

# Read the data
event_df = pd.read_csv(events_ffp, index_col="evid")
event_df.index = event_df.index.values.astype(str)
sites_df = pd.read_csv(sites_ffp, index_col="sta")

min_lon, max_lon, min_lat, max_lat = region

# Filter event and sites for the region of interest
event_df = event_df.loc[
    (event_df.lon > min_lon)
    & (event_df.lon < max_lon)
    & (event_df.lat > min_lat)
    & (event_df.lat < max_lat)
]

sites_df = sites_df.loc[
    (sites_df.lon > min_lon)
    & (sites_df.lon < max_lon)
    & (sites_df.lat > min_lat)
    & (sites_df.lat < max_lat)
]

# Magnitude filter
event_df = event_df.loc[event_df.mag > min_mag]


# Load map data
map_data = (
    None
    if map_data_ffp is None
    else plotting.NZMapData.load(map_data_ffp, high_res_topo=use_high_res_topo)
)

# Generate the figure
fig = plotting.gen_region_fig(
    region=(min_lon, max_lon, min_lat, max_lat),
    map_data=map_data,
    config_options=dict(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xx",
        MAP_FRAME_PEN="thinner,black",
        FONT_ANNOT_PRIMARY="6p,Helvetica,black",
    ),
)

# Plot the events
for ix, (cur_event, cur_row) in enumerate(event_df.iterrows()):
    cur_c = (
        "green"
        if events_to_highlight is not None and str(cur_event) in events_to_highlight
        else "red"
    )
    fig.meca(
        spec=dict(
            strike=cur_row.strike,
            dip=cur_row.dip,
            rake=cur_row.rake,
            magnitude=cur_row.mag,
        ),
        scale=f"{0.06 * cur_row.mag}c",
        G=cur_c,
        W="0.05p,black,solid",
        longitude=cur_row.lon,
        latitude=cur_row.lat,
        depth=cur_row.depth,
    )

# Create the inset rectangle (on main plot)
fig.plot(
    data=[[inset_region[0], inset_region[2], inset_region[1], inset_region[3]]],
    style="r+s",
    pen="0.5p,black",
)

# Plot the sites
for ix, (cur_site, cur_row) in enumerate(sites_df.iterrows()):
    fig.plot(
        x=cur_row.lon,
        y=cur_row.lat,
        style="t0.25c",
        fill="darkblue",
        pen="0.1p,darkblue",
    )

    if sites_to_highlight is not None and cur_site in sites_to_highlight:
        # Draw circle around the site
        fig.plot(
            x=cur_row.lon,
            y=cur_row.lat,
            style="c0.4c",
            fill=None,
            pen="0.5p,magenta",
        )

# Create the inset
with fig.inset(
    position="jTR",  # +o0.2c",
    region=inset_region,
    projection="M4c",
    margin=0,
    box="+p0.5p,black",
):
    fig.basemap(frame=False)

    # Plots the default coast (sea & inland lakes/rivers)
    if map_data is None:
        fig.coast(
            shorelines=["1/0.1p,black", "2/0.1p,black"],
            resolution="f",
            land="#666666",
            water="skyblue",
        )
    # Use the custom NZ data
    else:
        plotting._draw_map_data(fig, map_data, plot_kwargs=plotting.DEFAULT_PLT_KWARGS)

    # Plot the sites
    inset_sites_df = sites_df.loc[
        (sites_df.lon > inset_region[0])
        & (sites_df.lon < inset_region[1])
        & (sites_df.lat > inset_region[2])
        & (sites_df.lat < inset_region[3])
    ]
    for ix, (cur_site, cur_row) in enumerate(inset_sites_df.iterrows()):
        fig.plot(
            x=cur_row.lon,
            y=cur_row.lat,
            style="t0.15c",
            fill="darkblue",
            pen="0.1p,black",
        )

    # Plot the site of interest
    fig.plot(
        x=172.636849,
        y=-43.530954,
        style="a0.2c",
        fill="orange",
        pen="0.1p,black",
    )
    fig.text(
        text="Site of Interest",
        x=172.64,
        y=-43.530954,
        justify="LM",
        font="6p,Helvetica,black",
    )

fig.savefig(
    output_ffp,
    dpi=900,
    anti_alias=True,
)
