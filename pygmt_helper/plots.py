import multiprocessing as mp
from importlib import reload
from pathlib import Path
from typing import Tuple, Union, Sequence, Dict

import numpy as np
import pandas as pd
from qcore import nhm

from . import plotting


def im_plot(
    data_df: pd.DataFrame,
    im: str,
    rupture_name: str,
    hypo_loc: Tuple[float, float] = None,
    fault_trace: np.ndarray = None,
    cb_limits: Tuple[float, float] = None,
    nz_map_data: plotting.NZMapData = None,
    region: Union[str, Tuple[float, float, float, float]] = None,
):
    """
    Creates a (single) IM plot figure for
    the given rupture

    Parameters
    ----------
    data_df: dataframe
        Must contain lat, lon and im column,
        where IM values are in logspace
    im: str
        Name of IM (and key into data_df)
    rupture_name: str
        Used in the title
    hypo_loc: pair of floats, optional
        Longitude and latitude of the hypocentre
    cb_limits: pair of floats, optional
        Colormap/bar limits, (min, max)
        Note: These need to be in IM-space (i.e. not logged)
    fault_trace: array of floats, optional
    nz_map_data: NZMapData, optional
        Required for plotting roads & topo


    Returns
    -------
    fig: Figure
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
        projection=f"M17.0c",
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
    region: Union[str, Tuple[float, float, float, float]] = (
        164.8,
        179.4,
        -47.5,
        -36.0,
    ),
    map_data: plotting.NZMapData = None,
    title: str = "Faults",
    show_hypo: bool = False,
    highlight_faults: Sequence[str] = None,
):
    """
    Creates a figure showing all
    fault traces and the hypocentre of historic events

    Parameters
    ----------
    rupture_df: Dataframe
        Contains the hypocentre locations
        Required columns: [hlon, hlat, historic]
        where historic indicates if its an historic event
        or a realisation
    fault_data: list of NHMFault
    region: str or tuple of floats
        Region of the plot, see gen_region_fig
        for full details
    map_data: NZMapData
        Custom map data from qcore
    title: str
        Title of the plot
    show_hypo: bool
        If true, then plot Cybershake hypocentres
    highlight_faults: list of str
        Highlight the given faults in red


    Returns
    -------
    fig: Figure
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
    ims: Union[Sequence[str], str],
    output_dir: Path,
    rel_name: str = None,
    nhm_ffp: Path = None,
    qcore_data_dir: Path = None,
    cb_limits_dict: Dict[str, Tuple[float, float]] = None,
    n_procs: int = 1,
):
    """Generates IM plots using the given data"""
    nz_map_data = (
        None if qcore_data_dir is None else plotting.NZMapData.load(qcore_data_dir)
    )

    # Load the trace if possible
    fault_trace = (
        qcore.nhm.load_nhm(str(nhm_ffp))[fault].trace if nhm_ffp is not None else None
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
    hypo_loc: Tuple[float, float],
    fault_trace: np.ndarray,
    cb_limits: Union[Tuple[float, float], None],
    out_ffp: Path,
    nz_map_data: plotting.NZMapData = None,
    mp: bool = False,
):
    """MP Helper function"""
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
