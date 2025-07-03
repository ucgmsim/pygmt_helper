"""Plotting functions for generating IM (Intensity Measure) maps and fault trace plots."""

import io
import multiprocessing as mp
from collections.abc import Sequence
from importlib import reload
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pygmt
from qcore import nhm
from tqdm import tqdm

from . import plotting


def im_plot(
    data_df: pd.DataFrame,
    im: str,
    rupture_name: str,
    hypo_loc: Optional[tuple[float, float]] = None,
    fault_trace: Optional[np.ndarray] = None,
    cb_limits: Optional[tuple[float, float]] = None,
    nz_map_data: Optional[plotting.NZMapData] = None,
    region: Optional[str | tuple[float, float, float, float]] = None,
) -> pygmt.Figure:
    """
    Creates a single intensity measure (IM) plot figure for a given rupture.

    Parameters
    ----------
    data_df : pd.DataFrame
        Dataframe containing columns `lat`, `lon`, and an IM column.
        IM values should be in log-space.
    im : str
        Name of the IM, also used as a key in `data_df`.
    rupture_name : str
        Name of the rupture, used in the plot title.
    hypo_loc : tuple[float, float], optional
        Longitude and latitude of the hypocenter.
    fault_trace : np.ndarray, optional
        Array representing the fault trace coordinates.
    cb_limits : tuple[float, float], optional
        Colormap/bar limits as (min, max).
        These should be in IM-space (i.e., not logged).
    nz_map_data : plotting.NZMapData, optional
        Map data required for plotting roads and topography.
    region : str or tuple[float, float, float, float], optional
        The map region to plot. If `None`, it is inferred from `data_df`.

    Returns
    -------
    pygmt.Figure
        A PyGMT figure object containing the IM plot.

    Notes
    -----
    - The function computes an interpolated grid of IM values and plots it.
    - The color map is automatically scaled unless `cb_limits` is provided.
    - Roads and topography are plotted if `nz_map_data` is supplied.
    - The hypocenter and fault trace, if provided, are overlaid on the figure.
    """
    region = (
        (
            data_df.lon.min(),
            data_df.lon.max(),
            data_df.lat.min(),
            data_df.lat.max(),
        )
        if region is None
        else region
    )

    fig = plotting.gen_region_fig(
        rupture_name,
        region,
        projection="M17.0c",
        plot_roads=True if nz_map_data is not None else False,
        plot_topo=True if nz_map_data is not None else False,
        map_data=nz_map_data,
    )

    # Apply exponential
    cur_df = data_df[["lon", "lat"]].copy()
    cur_df[im] = np.exp(data_df[im].values)

    grid = plotting.create_grid(cur_df, im, region=region)

    # Colormap/bar limits
    if cb_limits is None:
        cb_min, cb_max = (
            np.round(np.quantile(cur_df[im].values, 0.02), 3),
            np.round(np.quantile(cur_df[im].values, 0.98), 3),
        )
    else:
        cb_min, cb_max = cb_limits

    plotting.plot_grid(
        fig,
        grid,
        "hot",
        (cb_min, cb_max, np.round(np.abs(cb_max - cb_min) / 12, 5)),
        ("white", "black"),
        reverse_cmap=True,
        transparency=35,
        cb_label=im.IM.from_im_name(im).pretty_im_name,
    )

    if fault_trace is not None:
        fig.plot(x=fault_trace[:, 0], y=fault_trace[:, 1], pen="0.5p,blue")

    if hypo_loc is not None:
        fig.plot(
            x=hypo_loc[0],
            y=hypo_loc[1],
            style="a0.3c",
            color="red",
            pen="black",
        )

    return fig


def faults_plot(
    rupture_df: pd.DataFrame,
    fault_data: Sequence[nhm.NHMFault],
    region: str | tuple[float, float, float, float] = (164.8, 179.4, -47.5, -36.0),
    map_data: Optional[plotting.NZMapData] = None,
    title: str = "Faults",
    show_hypo: bool = False,
    highlight_faults: Optional[Sequence[str]] = None,
) -> pygmt.Figure:
    """
    Creates a figure showing all fault traces and the hypocenters of historic events.

    Parameters
    ----------
    rupture_df : pd.DataFrame
        Dataframe containing hypocenter locations.
        Required columns: `hlon`, `hlat`, `historic`, and `mag`.
        - `hlon`, `hlat` : Longitude and latitude of the hypocenter.
        - `historic` : Boolean indicating if the event is historic or a realization.
        - `mag` : Magnitude of the event (used for marker sizing).
    fault_data : Sequence[nhm.NHMFault]
        List of NHMFault objects representing fault traces.
    region : str or tuple[float, float, float, float], default=(164.8, 179.4, -47.5, -36.0)
        Geographic region for the plot. See `gen_region_fig` for details.
    map_data : plotting.NZMapData, optional
        Custom map data for roads and topography.
    title : str, default="Faults"
        Title of the plot.
    show_hypo : bool, default=False
        If `True`, plots Cybershake hypocenters (non-historic events).
    highlight_faults : Sequence[str], optional
        List of fault names to highlight in red.

    Returns
    -------
    pygmt.Figure
        A PyGMT figure object containing the fault trace plot.
    """
    fig = plotting.gen_region_fig(
        title,
        region,
        map_data=map_data,
        plot_topo=True,
        plot_roads=False,
        plot_highways=True,
        plot_kwargs={"highway_pen_width": 0.1, "coastline_pen_width": 0.01},
    )

    # Plot the historic events
    if np.any(rupture_df.historic):
        fig.plot(
            x=rupture_df.loc[rupture_df.historic].hlon,
            y=rupture_df.loc[rupture_df.historic].hlat,
            size=0.005 * (2 ** rupture_df.loc[rupture_df.historic].mag),
            style="cc",
            fill="white",
            pen="0.1p,black",
        )

    if highlight_faults is None:
        highlight_faults = []

    # Plot the fault traces
    for cur_fault in fault_data:
        cur_trace = cur_fault.trace
        fig.plot(
            x=cur_trace[:, 0],
            y=cur_trace[:, 1],
            pen="2.0p,red" if cur_fault.name in highlight_faults else "0.5p,black",
        )

    # Plot the cybershake hypocentres
    if show_hypo:
        fig.plot(
            x=rupture_df.loc[~rupture_df.historic].hlon,
            y=rupture_df.loc[~rupture_df.historic].hlat,
            style="x0.04c",
            color="black",
            pen="black",
        )

    return fig


def im_plots(
    data_df: pd.DataFrame,
    fault: str,
    ims: Sequence[str] | str,
    output_dir: Path,
    rel_name: Optional[str] = None,
    nhm_ffp: Optional[Path] = None,
    nz_map_data: Optional[plotting.NZMapData] = None,
    cb_limits_dict: Optional[dict[str, tuple[float, float]]] = None,
    n_procs: int = 1,
) -> None:
    """
    Generates IM (Intensity Measure) plots using the provided data.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing the data for generating the IM plots, including latitudes, longitudes, and intensity measure values.
    fault : str
        Name of the fault to be used for generating the plots.
    ims : Union[Sequence[str], str]
        A sequence of strings or a single string indicating the intensity measures for the plots.
    output_dir : Path
        The directory where the generated plots will be saved.
    rel_name : str, optional
        If provided, the function will generate plots for a specific realization (run) identified by this name.
    nhm_ffp : Path, optional
        Path to the NHM file, which can provide fault trace information.
    nz_map_data : Optional[NZMapData], default=None
        The NZMapData object to supply topographic data.
    cb_limits_dict : dict, optional
        A dictionary specifying the color bar limits for each intensity measure (IM). If None, limits will be automatically computed.
    n_procs : int, optional
        Number of processes to use for parallelizing the plot generation. If set to 1, the function will generate plots sequentially.

    Returns
    -------
    None
        This function saves the generated plots to the specified output directory and does not return any values.
    """

    # Load the trace if possible
    fault_trace = (
        nhm.load_nhm(str(nhm_ffp))[fault].trace if nhm_ffp is not None else None
    )

    # Generate the plot/s
    for im in ims:
        cur_out_dir = output_dir / fault / (im).replace(".", "p")
        cur_out_dir.mkdir(exist_ok=True, parents=True)

        # Single realisation
        if rel_name is not None:
            __gen_im_map(
                data_df.loc[data_df.rel == rel_name, [im, "lat", "lon"]],
                im,
                rel_name,
                tuple(data_df.loc[data_df.rel == rel_name, ["hlon", "hlat"]].iloc[0]),
                fault_trace,
                None if cb_limits_dict is None else cb_limits_dict[im],
                cur_out_dir / f"{im.replace('.', 'p')}_{rel_name}.png",
                nz_map_data=nz_map_data,
            )
        # Multiple realisations
        else:
            # Common colormap/bar limits
            if cb_limits_dict is None:
                cb_limits = (
                    np.round(np.quantile(np.exp(data_df[im].values), 0.02), 3),
                    np.round(np.quantile(np.exp(data_df[im].values), 0.98), 3),
                )
            else:
                cb_limits = cb_limits_dict[im]

            if n_procs == 1:
                for cur_rel in np.unique(data_df.rel):
                    __gen_im_map(
                        data_df.loc[data_df.rel == cur_rel, [im, "lat", "lon"]],
                        im,
                        cur_rel,
                        tuple(
                            data_df.loc[data_df.rel == cur_rel, ["hlon", "hlat"]].iloc[
                                0
                            ]
                        ),
                        fault_trace,
                        cb_limits,
                        cur_out_dir / f"{im.replace('.', 'p')}_{cur_rel}.png",
                        nz_map_data=nz_map_data,
                    )
            else:
                with mp.Pool(n_procs) as p:
                    p.starmap(
                        __gen_im_map,
                        [
                            (
                                data_df.loc[data_df.rel == cur_rel, [im, "lat", "lon"]],
                                im,
                                cur_rel,
                                tuple(
                                    data_df.loc[
                                        data_df.rel == cur_rel, ["hlon", "hlat"]
                                    ].iloc[0]
                                ),
                                fault_trace,
                                cb_limits,
                                cur_out_dir / f"{im.replace('.', 'p')}_{cur_rel}.png",
                                nz_map_data,
                                True,
                            )
                            for cur_rel in np.unique(data_df.rel)
                        ],
                    )


def __gen_im_map(
    data_df: pd.DataFrame,
    im: str,
    rel_name: str,
    hypo_loc: tuple[float, float],
    fault_trace: np.ndarray,
    cb_limits: Optional[tuple[float, float]],
    out_ffp: Path,
    nz_map_data: plotting.NZMapData = None,
    mp: bool = False,
):
    """
    MP Helper function to generate an IM (Intensity Measure) map.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing the data for generating the IM plot, including intensity measure values, latitudes, and longitudes.
    im : str
        The intensity measure for which the map will be generated.
    rel_name : str
        The name of the realization (run) for which the map is generated.
    hypo_loc : tuple of float
        A tuple representing the hypocenter location in (longitude, latitude) coordinates.
    fault_trace : np.ndarray
        Array representing the fault trace data.
    cb_limits : tuple of float, optional
        Tuple of (min, max) values for the color bar limits. If None, the limits will be automatically determined.
    out_ffp : Path
        Path to the output file where the generated map will be saved.
    nz_map_data : plotting.NZMapData, optional
        Data for New Zealand map generation. If None, no map will be overlaid.
    mp : bool, optional
        If True, enables multiprocessing features.

    Returns
    -------
    None
        The function saves the generated IM map as a file at the specified output path.
    """
    if mp:
        import pygmt

        reload(pygmt)

    fig = im_plot(
        data_df,
        im,
        rel_name,
        cb_limits=cb_limits,
        fault_trace=fault_trace,
        nz_map_data=nz_map_data,
        hypo_loc=hypo_loc,
    )
    fig.savefig(str(out_ffp), dpi=1200)


def disagg_plot(
    disagg_df: pd.DataFrame,
    dist_mag_region: tuple[float, float, float, float],
    category_key: str,
    category_specs: dict,
    output_ffp: Path,
    plot_kwargs: dict | None = None,
) -> None:
    """
    Creates a 3D disaggregation plot.

    Parameters
    ----------
    disagg_df : pd.DataFrame
        DataFrame containing disaggregation data
        One row per magnitude/distance/category bin.
        Required columns:
        - 'mag': Magnitude of the earthquake.
        - 'dist': Rupture distance in km.
        - 'contribution': Contribution to hazard in percentage.
        - 'dist_bin_width': Width of the distance bin.
        - 'mag_bin_width': Width of the magnitude bin.
        - category_key: Category type (e.g. tectonic type or epsilon bin)
    dist_mag_region : tuple[float, float, float, float]
        A tuple defining the region for the plot in the format
        (min_distance, max_distance, min_magnitude, max_magnitude).
        This defines the x and y axes of the plot.
    category_key : str
        The column name in `disagg_df` that specifies the category type,
        e.g. epsilon or tectonic type.
    category_specs : dict
        A dictionary mapping category types to their specifications.
        Each key corresponds to a category type in `disagg_df[category_key]`,
        and each value is a tuple containing:
        - Name of the category (or None for default name).
        - Color specification for the category (e.g. "blue", "#FF0000").

        Note: There has to be one entry in this dictionary for each unique value
        in `disagg_df[category_key]`. 
    output_ffp : Path
        The file path where the plot will be saved.
    plot_kwargs : dict, optional
        Additional keyword arguments for the plot.
        Default values are used if not provided.
        Valid keys include:
        - 'width_factor': Factor to scale the width of the bin columns, 
            value has to be between 0 - 1. Default is 0.8.
        - 'zsize': Size of the z-axis in plot units (e.g. "5c" for 5 cm).
            Default is "5c".
        - 'perspective': Perspective of the 3D plot, given as a list of two angles
            [azimuth, elevation]. Default is [150, 35].
    """
    if len(category_specs) != disagg_df[category_key].nunique():
        raise ValueError(
            f"Category specifications do not match unique values in '{category_key}' column. "
            f"Expected {disagg_df[category_key].nunique()} unique values, got {len(category_specs)}."
        )

    # Determine maximum value for z-axis (contribution) and tick intervals
    max_contribution = np.ceil(disagg_df["contribution"].max() / 5) * 5
    z_major_ticks_interval = max_contribution / 5
    z_minor_ticks_interval = max_contribution / 10

    # Sort, to ensure correct plotting order
    disagg_df = disagg_df.sort_values(["mag", "dist"], ascending=(False, True))
    disagg_bin_groups = disagg_df.groupby(["mag", "dist"], sort=False)

    DEFAULT_PLOT_KWARGS = {
        "width_factor": 0.8,
        "zsize": "5c",
        "perspective": [150, 35],
    }
    plot_kwargs = (
        DEFAULT_PLOT_KWARGS
        if plot_kwargs is None
        else DEFAULT_PLOT_KWARGS | plot_kwargs
    )

    # Create the figure
    region = [
        dist_mag_region[0],
        dist_mag_region[1],
        dist_mag_region[2],
        dist_mag_region[3],
        0,
        max_contribution,
    ]
    fig = pygmt.Figure()
    fig.basemap(
        region=region,
        perspective=plot_kwargs["perspective"],
        # zscale=plot_kwargs["zscale"],
        zsize=plot_kwargs["zsize"],
        frame=[
            "wSnEZ1",
            "xa50f25+lRupture Distance (km)",
            "ya0.5f0.25+lMagnitude",
            f"za{z_major_ticks_interval}f{z_minor_ticks_interval}g{z_major_ticks_interval}+lContribution (%)",
        ],
    )

    # Iterate over mag/dist groups
    for cur_key, cur_group in tqdm(disagg_bin_groups):

        # Iterate over the category types
        cur_base = 0
        for _, cur_row in cur_group.iterrows():
            if np.isclose(cur_row["contribution"], 0.0):
                continue

            fig.plot3d(
                x=cur_row["dist"],
                y=cur_row["mag"],
                z=cur_base + cur_row["contribution"],
                region=region,
                perspective=plot_kwargs["perspective"],
                # zscale= plot_kwargs["zscale"],
                zsize=plot_kwargs["zsize"],
                style=f"o{cur_row['dist_bin_width'] * plot_kwargs['width_factor']}q/{cur_row['mag_bin_width'] * plot_kwargs['width_factor']}qb{cur_base}",
                fill=category_specs[cur_row[category_key]][1],
                pen="black",
            )

            cur_base += cur_row["contribution"]

    # Create legend specification using StringIO
    legend_spec = io.StringIO()
    legend_spec.write("H 14p,Helvetica-Bold Tectonic Region Type\n")
    legend_spec.write("D 0.1i 1p\n")
    for k, (name, color) in category_specs.items():
        name = k if name is None else name
        legend_spec.write(f"S 0.1i s 0.2i {color} 0.5p 0.3i {name}\n")

    fig.legend(
        spec=legend_spec,
        position="n1.0/1.0",
        box="+gwhite+p1p",
    )

    fig.savefig(output_ffp, dpi=900, anti_alias=True)
