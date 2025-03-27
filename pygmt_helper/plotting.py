"""Plotting tools to generate basemaps, grids and contours with PyGMT."""

import copy
import tempfile
from pathlib import Path
from typing import Any, NamedTuple, Optional, Self

import geopandas
import numpy as np
import pandas as pd
import pygmt
import xarray as xr
from qcore import gmt
from scipy import interpolate
from shapely import geometry


class NZMapData(NamedTuple):
    """New Zealand map data configuration."""

    road_df: pd.DataFrame = None
    highway_df: geopandas.GeoDataFrame = None
    coastline_df: geopandas.GeoDataFrame = None
    water_df: geopandas.GeoDataFrame = None
    topo_grid: xr.DataArray = None
    topo_shading_grid: xr.DataArray = None

    @classmethod
    def load(cls, high_res_topo: bool = False) -> Self:
        """Load NZMapData from qcore resources.

        Parameters
        ----------
        high_res_topo : bool
            If True, load high resolution topographic data.

        Returns
        -------
        NZMapData
            A map data object containing the paths to the map data.
        """
        road_ffp = gmt.GMT_DATA.fetch("data/Paths/road/NZ.gmt")
        highway_ffp = gmt.GMT_DATA.fetch("data/Paths/highway/NZ.gmt")
        coastline_ffp = gmt.GMT_DATA.fetch("data/Paths/coastline/NZ.gmt")
        water_ffp = gmt.GMT_DATA.fetch("data/Paths/water/NZ.gmt")

        if high_res_topo:
            topo_ffp = gmt.GMT_DATA.fetch("data/Topo/srtm_NZ_1s.grd")
            topo_shading_ffp = gmt.GMT_DATA.fetch("data/Topo/srtm_NZ_1s_i5.grd")
        else:
            topo_ffp = gmt.GMT_DATA.fetch("data/Topo/srtm_NZ.grd")
            topo_shading_ffp = gmt.GMT_DATA.fetch("data/Topo/srtm_NZ_i5.grd")

        return cls(
            road_df=geopandas.read_file(road_ffp),
            highway_df=geopandas.read_file(highway_ffp),
            coastline_df=geopandas.read_file(coastline_ffp),
            water_df=geopandas.read_file(water_ffp),
            topo_grid=pygmt.grdclip(grid=str(topo_ffp), below=[0.1, np.nan]),
            topo_shading_grid=pygmt.grdclip(
                grid=str(topo_shading_ffp), below=[0.1, np.nan]
            ),
        )


DEFAULT_PLT_KWARGS = dict(
    road_pen_width=0.01,
    highway_pen_width=0.5,
    coastline_pen_width=0.05,
    topo_cmap="gray",
    topo_cmap_min=-3000,
    topo_cmap_max=3000,
    topo_cmap_inc=10,
    topo_cmap_reverse=True,
    frame_args=["af", "xaf+lLongitude", "yaf+lLatitude"],
)


def gen_region_fig(
    title: Optional[str] = None,
    region: str | tuple[float, float, float, float] = "NZ",
    projection: str = "M17.0c",
    map_data: NZMapData = None,
    plot_roads: bool = True,
    plot_highways: bool = True,
    plot_topo: bool = True,
    plot_kwargs: dict[str, Any] = None,
    config_options: dict[str, str | int] = None,
    subtitle: Optional[str] = None,
):
    """
    Generates a basic map figure for a specified region, including coastlines,
    roads, highways, and topography if specified.

    Parameters
    ----------
    title : str, optional
        Title of the figure.
    region : str or tuple of float
        The region to plot. If a string, it should correspond to a predefined region
        (e.g., "NZ" for New Zealand). If a tuple, it must be in the format
        (min_lon, max_lon, min_lat, max_lat).
    projection : str
        The map projection string. See the PyGMT documentation [0]_ for details.
    map_data : NZMapData, optional
        Custom map data object from qcore, used to plot additional geographical
        features.
    plot_roads : bool, optional, default=True
        If True, plot roads on the map.
    plot_highways : bool, optional, default=True
        If True, plot highways on the map.
    plot_topo : bool, optional, default=True
        If True, plot topography on the map.
    plot_kwargs : dict, optional
        Additional plotting arguments, overriding values in `DEFAULT_PLT_KWARGS`.
        Only specify the options to be overridden.
    config_options : dict, optional
        Configuration options to apply to the figure. See the GMT configuration
        documentation [1]_ for available options.
    subtitle : str, optional
        Subtitle of the figure.

    Returns
    -------
    pygmt.Figure
        A PyGMT figure object representing the generated map.

    References
    ----------
    .. [0] https://www.pygmt.org/latest/projections/index.html#projections
    .. [1] https://docs.generic-mapping-tools.org/latest/gmt.conf.html
    """
    # Merge with default
    plot_kwargs = copy.deepcopy(DEFAULT_PLT_KWARGS) | (plot_kwargs or {})

    if title:
        plot_kwargs["frame_args"] = plot_kwargs.get("frame_args", []) + [
            f"+t{title}".replace(" ", r"\040")
        ]

    if subtitle:
        plot_kwargs["frame_args"] = plot_kwargs.get("frame_args", []) + [
            f"+s{subtitle}".replace(" ", r"\040")
        ]

    fig = pygmt.Figure()

    if config_options:
        pygmt.config(**config_options)

    fig.basemap(region=region, projection=projection, frame=plot_kwargs["frame_args"])

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
        _draw_map_data(
            fig,
            map_data,
            plot_topo=plot_topo,
            plot_roads=plot_roads,
            plot_highways=plot_highways,
            plot_kwargs=plot_kwargs,
        )

    return fig


def _draw_map_data(
    fig: pygmt.Figure,
    map_data: Optional[NZMapData],
    plot_topo: bool = True,
    plot_roads: bool = True,
    plot_highways: bool = True,
    plot_kwargs: dict[str, str | int] = None,
) -> None:
    """Draws map data on a PyGMT figure.

    Parameters
    ----------
    fig : pygmt.Figure
        The PyGMT figure to draw on.
    map_data : Optional[NZMapData]
        The map data containing coastline, topography, water, and road networks.
    plot_topo : bool, optional
        Whether to plot topographic data. Defaults to True.
    plot_roads : bool, optional
        Whether to plot roads. Defaults to True.
    plot_highways : bool, optional
        Whether to plot highways. Defaults to True.
    plot_kwargs : dict[str, str | int], optional
        A dictionary of plotting options, including:

        - ``"coastline_pen_width"`` (str or int): Line width for coastline.
        - ``"topo_cmap_min"`` (int): Minimum value for the topography colormap.
        - ``"topo_cmap_max"`` (int): Maximum value for the topography colormap.
        - ``"topo_cmap_inc"`` (int): Increment for the topography colormap.
        - ``"topo_cmap"`` (str): Name of the colormap for topography.
        - ``"topo_cmap_reverse"`` (bool): Whether to reverse the topography colormap.
        - ``"road_pen_width"`` (str or int): Line width for roads.
        - ``"highway_pen_width"`` (str or int): Line width for highways.
    """
    # Plot coastline and background water
    water_bg = geopandas.GeoSeries(
        geometry.LineString(
            [
                (fig.region[0], fig.region[2]),
                (fig.region[1], fig.region[2]),
                (fig.region[1], fig.region[3]),
                [fig.region[0], fig.region[3]],
            ]
        )
    )
    fig.plot(water_bg, fill="lightblue", straight_line=True)
    fig.plot(
        data=map_data.coastline_df,
        pen=f"{plot_kwargs['coastline_pen_width']}p,black",
        fill="lightgray",
    )

    # Add topo
    if plot_topo:
        pygmt.makecpt(
            series=(
                plot_kwargs["topo_cmap_min"],
                plot_kwargs["topo_cmap_max"],
                plot_kwargs["topo_cmap_inc"],
            ),
            continuous=False,
            cmap=plot_kwargs["topo_cmap"],
            reverse=plot_kwargs["topo_cmap_reverse"],
        )
        fig.grdimage(
            grid=map_data.topo_grid,
            shading=map_data.topo_shading_grid,
            cmap=True,
            nan_transparent=True,
        )

    # Plot water
    fig.plot(data=map_data.water_df, fill="lightblue")

    # Add roads
    if plot_roads:
        fig.plot(data=map_data.road_df, pen=f"{plot_kwargs['road_pen_width']}p,white")
    if plot_highways:
        fig.plot(
            data=map_data.highway_df,
            pen=f"{plot_kwargs['highway_pen_width']}p,yellow",
        )


def plot_grid(
    fig: pygmt.Figure,
    grid: xr.DataArray,
    cmap: str,
    cmap_limits: tuple[float, float, float],
    cmap_limit_colors: tuple[str, str],
    cb_label: str = None,
    reverse_cmap: bool = False,
    log_cmap: bool = False,
    transparency: float = 0.0,
    plot_contours: bool = True,
    continuous_cmap: bool = False,
):
    """Plots a data grid as a color map with optional contours and a color bar.

    Parameters
    ----------
    fig : pygmt.Figure
        The PyGMT figure object to plot on.
    grid : xr.DataArray
        The data grid to be plotted. The grid must have latitude and
        longitude coordinates (in that order) along with associated
        data values.
    cmap : str
        The name of the master colormap to use. See GMT documentation
        for available colormaps [0]_.
    cmap_limits : tuple of (float, float, float)
        The minimum, maximum, and step values for the colormap. The
        number of colors is determined by:
        ```
        (cmap_limits[1] - cmap_limits[0]) / cmap_limits[2]
        ```
    cmap_limit_colors : tuple of (str, str)
        Colors to use for data values outside the specified colormap
        range. The first color is for values below the minimum, and
        the second is for values above the maximum.
    cb_label : str, optional
        Label for the color bar.
    reverse_cmap : bool, optional, default=False
        If True, reverses the colormap.
    log_cmap : bool, optional, default=False
        If True, applies a logarithmic (base-10) scale to the colormap.
        Expects `cmap_limits` to be in log10 scale.
    transparency : float, optional, default=0.0
        Transparency level of the color map (0-100).
    plot_contours : bool, optional, default=True
        If True, adds contour lines at every second colormap step.
    continuous_cmap : bool, optional, default=False
        If True, generates a continuous colormap instead of a discrete
        one. See `pygmt.makecpt`.

    Returns
    -------
    None
        The function modifies the provided PyGMT figure in place.

    References
    ----------
    .. [0] https://docs.generic-mapping-tools.org/latest/cookbook/cpts.html.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Set the background & foreground colour for the colormap
        pygmt.config(
            COLOR_BACKGROUND=cmap_limit_colors[1], COLOR_FOREGROUND=cmap_limit_colors[0]
        )

        # Need two CPTs, otherwise the contours will be plotted every cb_step
        # And using "interval" directly in the contour call means that they don't
        # line up with the colour map
        cpt_ffp, cpt_ffp_ct = (
            str(tmp_dir / "cur_cpt_1.cpt"),
            str(tmp_dir / "cur_cpt_2.cpt"),
        )
        pygmt.makecpt(
            cmap=cmap,
            series=[cmap_limits[0], cmap_limits[1], cmap_limits[2]],
            output=cpt_ffp,
            reverse=reverse_cmap,
            log=log_cmap,
            continuous=continuous_cmap,
        )
        pygmt.makecpt(
            cmap=cmap,
            series=[cmap_limits[0], cmap_limits[1], cmap_limits[2] * 2],
            output=cpt_ffp_ct,
            reverse=reverse_cmap,
            log=log_cmap,
        )

        # Plot the grid
        fig.grdimage(
            grid,
            cmap=cpt_ffp,
            transparency=transparency,
            interpolation="c",
            nan_transparent=True,
        )

        # Plot the contours
        if plot_contours:
            fig.grdcontour(
                annotation="-",
                interval=cpt_ffp_ct,
                grid=grid,
                limit=[cmap_limits[0], cmap_limits[1]],
                pen="0.1p",
            )

        # Add a colorbar, with an annotated tick every second colour step,
        # and un-annotated tick with every other colour step
        phase = f"+{cmap_limits[0]}" if cmap_limits[0] > 0 else f"+{cmap_limits[1]}"
        cb_frame = [f"a+{cmap_limits[2] * 2}{phase}f+{cmap_limits[2]}"]
        if cb_label is not None:
            cb_frame.append(f"x+l{cb_label.replace(' ', r'\040')}")
        fig.colorbar(
            cmap=cpt_ffp,
            frame=cb_frame,
        )


def create_grid(
    data_df: pd.DataFrame,
    data_key: str,
    grid_spacing: str = "200e/200e",
    region: str | tuple[float, float, float, float] = "NZ",
    interp_method: str = "linear",
    set_water_to_nan: bool = True,
):
    """Generates a regular grid from unstructured data.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Unstructured data to be gridded. The DataFrame must contain the following
        columns: `lon` (longitude), `lat` (latitude), and a data value column
        (referred to as `data_key`).
    data_key : str
        The column name in `data_df` containing the data values.
    grid_spacing : str
        Grid spacing specification, using GMT gridding conventions.
        See the spacing parameter of `pygmt.grdlandmask` or the Notes section.
    region : str or tuple of (float, float, float, float)
        The region to plot. If a string, it should correspond to a predefined region
        (e.g., "NZ" for New Zealand). If a tuple, it must be in the format
        (min_lon, max_lon, min_lat, max_lat).
    interp_method : str
        The interpolation method to apply between points in `data_df`. Must be one of `"CloughTorcher"`, `"nearest"` or `"linear"`.
    set_water_to_nan : bool
        If True, set water values in the grid to NaN.

    Returns
    -------
    xarray.DataArray
        A gridded representation of the input data.

    Notes
    -----
    Common grid spacing formats:
    - To specify grid spacing of `x` units: `"{x}{unit}/{x}{unit}"`,
      where `unit` can be metres (`e`) or kilometres (`k`).
    - To define a fixed number of gridlines: `"{x}+n/{x}+n"`,
      where `x` is the number of gridlines.
    """

    # Create the land/water mask
    land_mask = pygmt.grdlandmask(
        region=region, spacing=grid_spacing, maskvalues=[0, 1, 1, 1, 1], resolution="f"
    )

    # Use land/water mask to create meshgrid
    x1, x2 = np.meshgrid(land_mask.lon.values, land_mask.lat.values)

    # Interpolate available data onto meshgrid
    if interp_method == "CloughTorcher":
        interp = interpolate.CloughTocher2DInterpolator(
            np.stack((data_df.lon.values, data_df.lat.values), axis=1),
            data_df[data_key].values,
        )
    elif interp_method == "nearest":
        interp = interpolate.NearestNDInterpolator(
            np.stack((data_df.lon.values, data_df.lat.values), axis=1),
            data_df[data_key].values,
        )
    elif interp_method == "linear":
        interp = interpolate.LinearNDInterpolator(
            np.stack((data_df.lon.values, data_df.lat.values), axis=1),
            data_df[data_key].values,
        )
    else:
        raise ValueError(
            "Invalid interpolation method specified, "
            "has to be one of [CloughTorcher, nearest, linear]"
        )

    grid_values = interp(x1, x2)

    # Create XArray grid
    grid = xr.DataArray(
        grid_values.reshape(land_mask.lat.size, land_mask.lon.size).astype(float),
        dims=("lat", "lon"),
        coords={"lon": np.unique(x1), "lat": np.unique(x2)},
    )

    # Change water values to nan
    if set_water_to_nan:
        grid.values[~land_mask.astype(bool)] = np.nan

    return grid
