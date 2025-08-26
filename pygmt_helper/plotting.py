"""Plotting tools to generate basemaps, grids and contours with PyGMT."""

import itertools
import copy
import tempfile
from pathlib import Path
from typing import Any, NamedTuple, Optional, Self, Callable

import geopandas
import numpy as np
import pandas as pd
import pooch
import pygmt
import xarray as xr
from scipy import interpolate
from qcore import point_in_polygon

GMT_DATA = pooch.create(
    pooch.os_cache("pygmt_helper"),
    base_url="",
    registry={
        "data/Paths/water/NZ.gmt": "sha256:9abdd22ee120ce50613d3745825caeac5fc6f9ccec3bc80a4bc33d6de6cbd218",
        "data/Paths/water/NZ.parquet": "sha256:1ede5670a17c0a8cadef7de352d47b59b36d90ffab11ad8cc6d1e2d2eddbbe6a",
        "data/Topo/srtm_KR.grd": "sha256:cc59be8e9ee8cabb75587c040fd67597eb02116e225eeae89949e6f924058325",
        "data/Topo/srtm_NZ.grd": "sha256:adb3eb43cd20be468b15cba52f8953538bf7523361f1f2d7b68dbf74113cc06c",
        "data/Paths/water/KR.gmt": "sha256:9950b917d3f4e239e908f93f65705424082ae55f072d6a7926bb56298c2f5b28",
        "data/Topo/srtm_KR_i5.grd": "sha256:adbacea622607b91438fee68999ebc7c8dd9eb35b3230708a9a5a21fc0de472b",
        "data/regions.ll": "sha256:17ad7202395af54dea08f93f0b9ed8438fcb05834bc12242fa4fb770395ba899",
        "data/Paths/coastline/NZ.gmt": "sha256:31660def8f51d6d827008e6f20507153cfbbfbca232cd661da7f214aff1c9ce3",
        "data/Paths/coastline/NZ.parquet": "sha256:8106ffdf1c5d826acede353e49011a8ff1bc490881bda99acd880c9b2dcdb5d2",
        "data/Paths/highway/NZ.gmt": "sha256:fd03908ecd137fa0bd20184081d7b499d23bc44e4154dad388b3ba8c89893e62",
        "data/Paths/highway/NZ.parquet": "sha256:e397a5d5f9662abf0627d05ed79dc21ab975546acc3c1d7668d7904c7fd735d3",
        "data/version": "sha256:44804414f85bef9588f60086587fd6e8871b39123c831ec129624f4d81a95fea",
        "data/cpt/nz_topo_grey1.cpt": "sha256:39305ac0739757337241a602a2dca71d0981a9fcc0e3240388b078669f1b3f84",
        "data/cpt/hot-orange-log.cpt": "sha256:c56a2b43690468753489ff56817197ef7faab456a979c2dd9bb6bab80947dc14",
        "data/cpt/slip.cpt": "sha256:e243f96aad43ea58fb0a1ed4c566d1e5d587abaf286a367dcd2be60a395dfc28",
        "data/Paths/highway/KR.gmt": "sha256:bf2cbc7efd7e6fb8d3265ed421eda61fbe768fc6ddc5ed0f5c8f06ece023f909",
        "data/cpt/hot-orange.cpt": "sha256:dace12cae5d803a4842af83e1ebee151cd797fede9238e1860574423a3aa7838",
        "data/cpt/liquefaction_susceptibility.cpt": "sha256:29fb2b4e0fca678d5c28ad49d34411a0c411257b0843c94fc27ad23bfe4030cf",
        "data/cpt/palm_springs_nz_topo.cpt": "sha256:8bb174d0fb86ea0181e8216cb75c04128aec29121aa1eae6a65344c4c84884b1",
        "data/cpt/mmi.cpt": "sha256:4607b77a230b2ff33f8ff700ddd502df1c4c3604af01c64215d699e81bea5590",
        "data/cpt/trise.cpt": "sha256:3711884ab8a216f102a1f60cdc4cfbb1aca3f3ab54fb08f1ae768eda77b88047",
        "data/cpt/landslide_susceptibility.cpt": "sha256:1dbf19be72e42181da0f60d8a081c97a29da64b1473ae10f011061532dad218f",
        "data/cpt/landslide_susceptibility_nolabel.cpt": "sha256:936d8f7cebb34e91ceff2f02a8a14923db280367020a879f9861584703f63e64",
        "data/cpt/mmi-labels.cpt": "sha256:9b93ccfa22a3719eae931423a0fe67fa91dbbcfda008b792d136ecf07f0deffe",
        "data/cpt/palm_springs_1.cpt": "sha256:487694eecd04dbc90619a3fa156a971c000efe364d4bd9d808ef5cde00c7e773",
        "data/cpt/liquefaction_susceptibility_nolabel.cpt": "sha256:e0850f6c0a0614b95d77e0df195574fdff3f68e9c1f9c84642f8985a9cba92ca",
        "data/img/logo-right.png": "sha256:849b332b7d234a3508cf5393555d3d526097df2dcabd35df50055ac0022dbb4d",
        "data/img/logo-left.png": "sha256:e254ee4ca2c628e673b6ce04bd1f479d707493aab036e9a012c05f94b999ffdd",
        "data/Paths/road/KR.gmt": "sha256:99a3d6f0da95698c38dfa40e509f125f2713633612ceb2a52cf7286fa2c68358",
        "data/Paths/road/NZ.gmt": "sha256:e01f2ac2fc4a406e1d430c2cffb2d3ef10e260b10148fd9dc92723888cc24a68",
        "data/Paths/road/NZ.parquet": "sha256:046d8072a5ad4d45cb8f5123e3f5368ea0afb3fdd1b1f96c3fc82271146af41a",
        "data/Topo/srtm_NZ_i5.grd": "sha256:a2bd8c148015933b845a9760559457bd42b937fdd34ecb2d72a44f25e691cae4",
        "data/Topo/srtm_NZ_1s.grd": "sha256:1caecfefda5bf7906593dacc76eeb91123b1768d50b6fe4e3b8ee90a1a3bcdc6",
        "data/Topo/srtm_NZ_1s_i5.grd": "sha256:9a87328e680608542b49f719d230fb92c4a6a3b110720df50c2a6ad3b6c0547f",
    },
    # Now specify custom URLs for some of the files in the registry.
    urls={
        "data/Paths/coastline/NZ.gmt": "https://www.dropbox.com/scl/fi/zkohh794y0s2189t7b1hi/NZ.gmt?rlkey=02011f4morc4toutt9nzojrw1&st=vpz2ri8x&dl=1",
        "data/Paths/coastline/NZ.parquet": "https://www.dropbox.com/scl/fi/jpfw1ia678si1mlacv4js/NZ.parquet?rlkey=5mpkg8tzqs3ahfe6h5tr9odh6&st=i280m74j&dl=1",
        "data/Paths/water/NZ.gmt": "https://www.dropbox.com/scl/fi/ik101lnpkn3nn6z01ckcw/NZ.gmt?rlkey=byghec0ktpj00ctgau6704rl7&st=ng70q2fz&dl=1",
        "data/Paths/water/NZ.parquet": "https://www.dropbox.com/scl/fi/sfbkkeppcl45ypq0dqx65/NZ.parquet?rlkey=nanwv1qbva5zrq7ge82wmabco&st=u2bab0eh&dl=1",
        "data/Paths/water/KR.gmt": "https://www.dropbox.com/scl/fi/gwpr5ai97bx905qmaamvb/KR.gmt?rlkey=hw9bup7u1i0p4wog91vxdwkaz&st=8jxpkhyu&dl=1",
        "data/Paths/road/NZ.gmt": "https://www.dropbox.com/scl/fi/xu4o7gh4fd1nlolqr5kb2/NZ.gmt?rlkey=2h95i3sib6j1tjo6l4p14mlf7&st=6k1c1r5e&dl=1",
        "data/Paths/road/NZ.parquet": "https://www.dropbox.com/scl/fi/77njl4qzn6slq9ojme574/NZ.parquet?rlkey=roezlj84bcz3drahc8220nr30&st=5r8k531y&dl=1",
        "data/Paths/road/KR.gmt": "https://www.dropbox.com/scl/fi/u1v08tnqfwl69kbqc6vp6/KR.gmt?rlkey=rie315iw8zdgpqclegbhdto60&st=jlbcqxhe&dl=1",
        "data/Paths/highway/NZ.gmt": "https://www.dropbox.com/scl/fi/pycl9rapaw4h8oapnk2zx/NZ.gmt?rlkey=jup637ec1kabfq57il8q2z52i&st=5jpaxeih&dl=1",
        "data/Paths/highway/NZ.parquet": "https://www.dropbox.com/scl/fi/9d1daa55o7kz2dklzu6zj/NZ.parquet?rlkey=9qb4bnb0zgnvb641qyvtxgkd9&st=3jo7604w&dl=1",
        "data/Paths/highway/KR.gmt": "https://www.dropbox.com/scl/fi/ogs9bwlq1qcmqkm73e7tr/KR.gmt?rlkey=eneeceqzmbifuyg2f5sdc1roc&st=hrenqhm4&dl=1",
        "data/Topo/srtm_NZ.grd": "https://www.dropbox.com/scl/fi/mq99chc3u9nl0cqvszadj/srtm_NZ.grd?rlkey=kypozxtqfenheqz0lv0w9j9ee&st=jhhht7q3&dl=1",
        "data/Topo/srtm_NZ_i5.grd": "https://www.dropbox.com/scl/fi/mdbtf90bq7gnmh9vzpd9u/srtm_NZ_i5.grd?rlkey=mztlms8huuacq1ygujpwo9zia&st=pkwb2wfe&dl=1",
        "data/Topo/srtm_NZ_1s.grd": "https://www.dropbox.com/scl/fi/z3nymvy41rrxctuxh16xl/srtm_NZ_1s.grd?rlkey=ja1hmecgz3dz6zcblua64sr8t&st=x09hn3pu&dl=1",
        "data/Topo/srtm_NZ_1s_i5.grd": "https://www.dropbox.com/scl/fi/avzaeu6zqbhp4xkfqwtrt/srtm_NZ_1s_i5.grd?rlkey=iyj82hsqyrv7w7x6o5t9191jo&st=3i48q15r&dl=1",
        "data/Topo/srtm_KR.grd": "https://www.dropbox.com/scl/fi/ds23toeh73uj4tyza86kd/srtm_KR.grd?rlkey=knz42nbdhw0ozkarc9izp6941&st=t1v7v572&dl=1",
        "data/Topo/srtm_KR_i5.grd": "https://www.dropbox.com/scl/fi/rtzfo07s6gjdm9xofdj6h/srtm_KR_i5.grd?rlkey=kjb0quk06z8npz13hsaizgn4i&st=a5ix7lgn&dl=1",
        "data/regions.ll": "https://www.dropbox.com/scl/fi/073atd0ebcrmob46a8yp5/regions.ll?rlkey=g54pfbd6jr25k24vm6ohgy6dq&st=1sgbox8p&dl=1",
        "data/cpt/trise.cpt": "https://www.dropbox.com/scl/fi/scn9qbp5g7eq6qparbr5c/trise.cpt?rlkey=a7my5euwoqoqyi3xu5340o1jt&st=3pcuy7hj&dl=1",
        "data/cpt/slip.cpt": "https://www.dropbox.com/scl/fi/e7jwxfpeneke7g6ay4gqi/slip.cpt?rlkey=8ouopksidlsx6yy9acejspodt&st=vnq4tehy&dl=1",
        "data/cpt/palm_springs_nz_topo.cpt": "https://www.dropbox.com/scl/fi/1thpu13lmwtwfrblgse75/palm_springs_nz_topo.cpt?rlkey=46wame3m05ae0yb3axfblmaqe&st=8qnrtd9s&dl=1",
        "data/cpt/palm_springs_1.cpt": "https://www.dropbox.com/scl/fi/lfbjuw68be2437n5w0t57/palm_springs_1.cpt?rlkey=upzukhcz4nb2s81f8nmy9ezk7&st=dv9aipum&dl=1",
        "data/cpt/nz_topo_grey1.cpt": "https://www.dropbox.com/scl/fi/32kmnru3gdxslcyarb5se/nz_topo_grey1.cpt?rlkey=yioo4il6rdbs520mapaniulr1&st=92gqx1jq&dl=1",
        "data/cpt/mmi.cpt": "https://www.dropbox.com/scl/fi/wjjnwzydtfcl5v485vffy/mmi.cpt?rlkey=jvq9z8qg49fwk1uohej4v8m6r&st=ztkq2yt2&dl=1",
        "data/cpt/mmi-labels.cpt": "https://www.dropbox.com/scl/fi/xg7i949rhtgeeqdeo6qd7/mmi-labels.cpt?rlkey=yklw07uwqjo2yn0580gwy544b&st=j4xvri1x&dl=1",
        "data/cpt/liquefaction_susceptibility.cpt": "https://www.dropbox.com/scl/fi/2ocuygxo9qqq6v33os1r6/liquefaction_susceptibility.cpt?rlkey=wkbvwjjsl7mpc09bg7tedmztf&st=1txd338v&dl=1",
        "data/cpt/liquefaction_susceptibility_nolabel.cpt": "https://www.dropbox.com/scl/fi/sv35h9tbtmk8oo3x6gv6a/liquefaction_susceptibility_nolabel.cpt?rlkey=hgzcvq1uwppch6n70ff22s16t&st=j327gq8d&dl=1",
        "data/cpt/landslide_susceptibility.cpt": "https://www.dropbox.com/scl/fi/k5903mjgablxkotvoscsy/landslide_susceptibility.cpt?rlkey=rzjjatnbht021tdwc7rswgtlu&st=69rr315q&dl=1",
        "data/cpt/landslide_susceptibility_nolabel.cpt": "https://www.dropbox.com/scl/fi/5qfrh1fv7bcscopnsttvp/landslide_susceptibility_nolabel.cpt?rlkey=tdc9xeay84k30r6s4ze1198nt&st=6n7htezq&dl=1",
        "data/cpt/hot-orange.cpt": "https://www.dropbox.com/scl/fi/5gfr9mtykrge2fy6h4jrb/hot-orange.cpt?rlkey=pnyx5864v5ym6fhv237esjwqa&st=q1l2bxmb&dl=1",
        "data/cpt/hot-orange-log.cpt": "https://www.dropbox.com/scl/fi/ggq31kcc5e5qdn6guihoe/hot-orange-log.cpt?rlkey=8z05lhwkqz5on0nji5yhms1gl&st=7hbbih07&dl=1",
    },
)


class NZMapData(NamedTuple):
    """New Zealand map data configuration."""

    road_df: pd.DataFrame = None
    highway_df: geopandas.GeoDataFrame = None
    coastline_df: geopandas.GeoDataFrame = None
    water_df: geopandas.GeoDataFrame = None
    topo_grid: xr.DataArray = None
    topo_shading_grid: xr.DataArray = None

    @classmethod
    def load(
        cls,
        region: tuple[float, float, float, float] = None,
        high_res_topo: bool = False,
    ) -> Self:
        """Load NZMapData.

        Parameters
        ----------
        region : tuple of float, optional
            The region to load the map data for.
            If None, loads the full NZ map data.
            Currently only applied to the topography data.
        high_res_topo : bool, optional
            If True, load high resolution topographic data.

        Returns
        -------
        NZMapData
            A map data object containing the paths to the map data.
        """
        road_ffp = GMT_DATA.fetch("data/Paths/road/NZ.parquet")
        highway_ffp = GMT_DATA.fetch("data/Paths/highway/NZ.parquet")
        coastline_ffp = GMT_DATA.fetch("data/Paths/coastline/NZ.parquet")
        water_ffp = GMT_DATA.fetch("data/Paths/water/NZ.parquet")

        if high_res_topo:
            topo_ffp = GMT_DATA.fetch("data/Topo/srtm_NZ_1s.grd")
            topo_shading_ffp = GMT_DATA.fetch("data/Topo/srtm_NZ_1s_i5.grd")
        else:
            topo_ffp = GMT_DATA.fetch("data/Topo/srtm_NZ.grd")
            topo_shading_ffp = GMT_DATA.fetch("data/Topo/srtm_NZ_i5.grd")

        topo_grid = xr.open_dataset(topo_ffp)["z"]
        if region:
            topo_grid = topo_grid.sel(
                lon=slice(region[0], region[1]),
                lat=slice(region[2], region[3]),
            )

        topo_shading = xr.open_dataset(topo_shading_ffp)["z"]
        if region:
            topo_shading = topo_shading.sel(
                lon=slice(region[0], region[1]),
                lat=slice(region[2], region[3]),
            )

        bbox = (region[0], region[2], region[1], region[3]) if region else None
        return cls(
            road_df=geopandas.read_parquet(road_ffp, bbox=bbox).set_crs("EPSG:4326"),
            highway_df=geopandas.read_parquet(highway_ffp, bbox=bbox).set_crs(
                "EPSG:4326"
            ),
            coastline_df=geopandas.read_parquet(coastline_ffp, bbox=bbox).set_crs(
                "EPSG:4326"
            ),
            water_df=geopandas.read_parquet(water_ffp, bbox=bbox).set_crs("EPSG:4326"),
            topo_grid=topo_grid,
            topo_shading_grid=topo_shading,
        )


DEFAULT_PLT_KWARGS = dict(
    water_color="lightblue",
    land_color="lightgray",
    road_pen_width=0.01,
    road_pen_color="white",
    highway_pen_width=0.5,
    highway_pen_color="yellow",
    coastline_pen_width=0.05,
    coastline_pen_color="black",
    topo_cmap="gray",
    topo_cmap_min=-3000,
    topo_cmap_max=3000,
    topo_cmap_inc=10,
    topo_cmap_continous=False,
    topo_cmap_reverse=True,
    frame_args=["af", "xaf+lLongitude", "yaf+lLatitude"],
)


def gen_region_fig(
    title: Optional[str] = None,
    region: tuple[float, float, float, float] | None = None,
    projection: str = "M17.0c",
    plot_roads: bool = False,
    plot_highways: bool = True,
    plot_topo: bool = True,
    high_res_topo: bool = False,
    plot_kwargs: dict[str, Any] = None,
    config_options: dict[str, Any] = None,
    subtitle: Optional[str] = None,
    fig: pygmt.Figure | None = None,
    high_quality: bool = False,
    custom_shading_fn: (
        Callable[[xr.DataArray, xr.DataArray], xr.DataArray] | None
    ) = None,
):
    """
    Generates a basic map figure for a specified region, including coastlines,
    roads, highways, and topography if specified.

    Parameters
    ----------
    title : str, optional
        Title of the figure.
    region : tuple of float, optional
        The region to plot, defined as a tuple of four floats
        (min_lon, max_lon, min_lat, max_lat).
        If None, then creates a NZ-wide map.
    projection : str
        The map projection string. See the PyGMT documentation [0]_ for details.
    plot_roads : bool, optional, default=True
        If True, plot roads on the map.
    plot_highways : bool, optional, default=True
        If True, plot highways on the map.
    plot_topo : bool, optional, default=True
        If True, plot topography on the map.
        Setting this to False will significantly speed up plotting
    high_res_topo : bool, optional
        If True, use high resolution topography data.
        Requires ``plot_topo`` to be True.
        Only useful when plotting small regions, e.g. cities,
        makes no difference for large regions or NZ-wide maps.
        Increases plotting time.
    plot_kwargs : dict, optional
        Additional plotting arguments, overriding values in `DEFAULT_PLT_KWARGS`.
        Only specify the options to be overridden.

        Plotting options are:
        - ``"water_color"`` (str): Color for water.
        - ``"land_color"`` (str): Color for land. This only has a visible effect
            for very zoomed in maps, where differences between coastline definition
            and the topo tiles are visible.
        - ``"coastline_pen_width"`` (str or int): Line width for coastline.
        - ``"coastline_pen_color"`` (str): Color for coastline.
        - ``"topo_cmap_min"`` (int): Minimum value for the topography colormap.
        - ``"topo_cmap_max"`` (int): Maximum value for the topography colormap.
        - ``"topo_cmap_inc"`` (int): Increment for the topography colormap.
        - ``"topo_cmap"`` (str): Name of the colormap for topography.
        - ``"topo_cmap_continous"`` (bool): Whether to use a continuous colormap.
        - ``"topo_cmap_reverse"`` (bool): Whether to reverse the topography colormap.
        - ``"road_pen_width"`` (str or int): Line width for roads.
        - ``"road_pen_color"`` (str): Color for roads.
        - ``"highway_pen_width"`` (str or int): Line width for highways.
        - ``"highway_pen_color"`` (str): Color for highways.
    config_options : dict, optional
        Configuration options to apply to the figure. See the GMT configuration
        documentation [1]_ for available options.
    subtitle : str, optional
        Subtitle of the figure.
    fig : pygmt.Figure, optional
        A PyGMT figure object to plot on. If None, a new figure is created.
    high_quality : bool, optional
        If True, produce highest quality map.
        Should only be used for small regions, e.g. cities,
        Does not make a difference for large regions or NZ-wide maps.

        Currently only affects the topo plotting,
        where it ensures that the topo grid is aligned
        with the coastline and water boundaries.

        Increases plotting time.
    custom_shading_fn : Callable, optional
        A custom function to modify the topography shading grid.
        Function takes two arguments: topo_grid and topo_shading_grid,
        both of type `xr.DataArray`, and should return a modified
        `xr.DataArray` for the shading grid.

        Only applied if `plot_topo` and `high_quality` are True.

    Returns
    -------
    pygmt.Figure
        A PyGMT figure object representing the generated map.

    References
    ----------
    .. [0] https://www.pygmt.org/latest/projections/index.html#projections
    .. [1] https://docs.generic-mapping-tools.org/latest/gmt.conf.html
    """
    # Load NZ map data
    map_data = NZMapData.load(region=region, high_res_topo=high_res_topo and plot_topo)

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

    config_options = {"COLOR_NAN": plot_kwargs["water_color"]} | (config_options or {})
    pygmt.config(**config_options)

    if fig is None:
        fig = pygmt.Figure()

    water_color = plot_kwargs["water_color"]
    plot_kwargs["frame_args"] = plot_kwargs.get("frame_args", []) + [f"+g{water_color}"]
    fig.basemap(
        region=region if region else "NZ",
        projection=projection,
        frame=plot_kwargs["frame_args"],
    )

    # Plot coastline
    fig.plot(
        data=map_data.coastline_df,
        pen=f"{plot_kwargs['coastline_pen_width']}p,{plot_kwargs['coastline_pen_color']}",
        fill=plot_kwargs["land_color"],
    )

    # Add topo
    if plot_topo:
        # Drop topo points not on land based on
        # the coastline and water data.
        if high_quality and region:
            topo_points = np.array(
                list(
                    itertools.product(
                        map_data.topo_grid.lat.values, map_data.topo_grid.lon.values
                    )
                )
            )
            topo_land_mask = on_land(map_data, topo_points, region=region).reshape(
                map_data.topo_grid.shape
            )
            topo_grid = map_data.topo_grid.where(topo_land_mask, np.nan)
            topo_shading_grid = map_data.topo_shading_grid.where(topo_land_mask, np.nan)

            if custom_shading_fn:
                topo_shading_grid = custom_shading_fn(topo_grid, topo_shading_grid)
        # Drop topo points not on land, based on
        # topo grid elevation.
        else:
            # Sanity check
            assert np.allclose(
                map_data.topo_grid.lon.values, map_data.topo_shading_grid.lon.values
            )
            assert np.allclose(
                map_data.topo_grid.lat.values, map_data.topo_shading_grid.lat.values
            )

            mask = map_data.topo_grid >= 0.1
            topo_grid = map_data.topo_grid.where(mask, np.nan)
            topo_shading_grid = map_data.topo_shading_grid.reindex_like(
                topo_grid, method="nearest", tolerance=1e-6
            )
            topo_shading_grid = topo_shading_grid.where(mask, np.nan)

        # Create topography colormap
        pygmt.makecpt(
            series=(
                plot_kwargs["topo_cmap_min"],
                plot_kwargs["topo_cmap_max"],
                plot_kwargs["topo_cmap_inc"],
            ),
            continuous=plot_kwargs["topo_cmap_continous"],
            cmap=plot_kwargs["topo_cmap"],
            reverse=plot_kwargs["topo_cmap_reverse"],
            # Some CPTs define their own COLOR_NAN, but we wish to use the
            # water colour
            no_bg=True,
        )

        # Plot topography
        fig.grdimage(
            grid=topo_grid,
            shading=topo_shading_grid,
            cmap=True,
        )

    # Plot inland water
    fig.plot(data=map_data.water_df, fill=plot_kwargs["water_color"])

    # Add roads
    if plot_roads:
        fig.plot(
            data=map_data.road_df,
            pen=f"{plot_kwargs['road_pen_width']}p,{plot_kwargs['road_pen_color']}",
        )
    if plot_highways:
        fig.plot(
            data=map_data.highway_df,
            pen=f"{plot_kwargs['highway_pen_width']}p,{plot_kwargs['highway_pen_color']}",
        )

    return fig


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
            cb_label = cb_label.replace(" ", r"\040")
            cb_frame.append(f"x+l{cb_label}")
        fig.colorbar(
            cmap=cpt_ffp,
            frame=cb_frame,
        )


def create_grid(
    data_df: pd.DataFrame,
    data_key: str,
    grid_spacing: str = "200e/200e",
    region: tuple[float, float, float, float] | None = None,
    interp_method: str = "linear",
    set_water_to_nan: bool = True,
    high_quality: bool = False,
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
    region : tuple of float, optional
        The region to generate grid for, defined as a tuple of four floats
        (min_lon, max_lon, min_lat, max_lat).
        If None, then create NZ-wide grid.
    interp_method : str
        The interpolation method to apply between points in `data_df`. Must be one of `"CloughTorcher"`, `"nearest"` or `"linear"`.
    set_water_to_nan : bool
        If True, set water values in the grid to NaN.
    high_quality : bool, optional
        If True, use NZ-specific land/water mask for gridding,
        instead of pygmt.grdlandmask.

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
    if high_quality and region is not None:
        land_mask = get_landmask(
            NZMapData.load(region=region),
            region,
            grid_spacing=grid_spacing,
        )
    else:
        land_mask = pygmt.grdlandmask(
            region=region if region else "NZ",
            spacing=grid_spacing,
            maskvalues=[0, 1, 1, 1, 1],
            resolution="f",
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


def in_region(region: tuple[float, float, float, float], points: np.ndarray):
    """
    Check if points are within the specified region.

    Parameters
    ----------
    region : tuple
        A tuple defining the region as (min_lon, max_lon, min_lat, max_lat).
    points : np.ndarray
        An array of points with shape (n_points, 2) where each row is [lon, lat].

    Returns
    -------
    np.ndarray
        A boolean array indicating whether each point is within the region.
    """
    min_lon, max_lon, min_lat, max_lat = region
    lon, lat = points[:, 0], points[:, 1]

    return (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)


def on_land(
    map_data: NZMapData,
    points: np.ndarray,
    region: tuple[float, float, float, float] | None = None,
):
    """
    Checks if the given points are on land based on the
    `map_data` coastline and water polygons.

    Parameters
    ----------
    map_data : NZMapData
        The map data containing coastline and water polygons.
    points : np.ndarray
        An array of points with shape (n_points, 2) where each row is [lat, lon].
    region : tuple of (float, float, float, float), optional
        The region of interest, providing this will
        significantly speed up the check.
        If None, checks all points against the
        full NZ coastline and water polygons (slow).

    Returns
    -------
    np.ndarray
        A boolean array of the same length as `points`, where True indicates
        that the point is on land and False indicates that it is in water.
    """
    # Load coastline and water polygons
    coast_polygon_arrays = [
        np.array(list(cur_poly.coords))
        for cur_poly in map_data.coastline_df.values.ravel().tolist()
    ]
    water_polygon_arrays = [
        np.array(list(cur_poly.coords))
        for cur_poly in map_data.water_df.values.ravel().tolist()
    ]

    # Filter by region
    if region is not None:
        coast_polygon_arrays = [
            cur_poly_coords
            for cur_poly_coords in coast_polygon_arrays
            if np.any(in_region(region, cur_poly_coords))
        ]
        water_polygon_arrays = [
            cur_poly_coords
            for cur_poly_coords in water_polygon_arrays
            if np.any(in_region(region, cur_poly_coords))
        ]

    mask = np.zeros(points.shape[0], dtype=bool)
    for coast_polygon in coast_polygon_arrays:
        mask |= point_in_polygon.is_inside_postgis_parallel(
            points[:, ::-1], coast_polygon
        )
    for water_polygon in water_polygon_arrays:
        mask = np.where(
            point_in_polygon.is_inside_postgis_parallel(points[:, ::-1], water_polygon),
            False,
            mask,
        )

    return mask


def get_landmask(
    map_data: NZMapData,
    region: tuple[float, float, float, float],
    grid_spacing: str = "25e/25e",
):
    """
    Create a land mask grid for the specified region using
    the given coastline and water data.

    Parameters
    ----------
    map_data : NZMapData
        The map data containing coastline and water polygons.
    region : tuple of (float, float, float, float)
        The region to create the land mask for, defined as
        (min_lon, max_lon, min_lat, max_lat).
    grid_spacing : str, optional
        The grid spacing for the land mask.
        Defaults to 25metres ("25e/25e").

    Returns
    -------
    xarray.DataArray
        A land mask grid where land points are set to 1
        and water points are set to NaN.
    """
    # Create a land mask grid for the specified region and grid spacing.
    land_mask = pygmt.grdlandmask(region=region, spacing=grid_spacing)
    land_mask[:] = 1

    # Get grid lat/lon values
    grid_points = np.array(
        list(itertools.product(land_mask.lat.values, land_mask.lon.values))
    )
    grid_points_mask = on_land(map_data, grid_points, region=region)
    land_mask[:] = grid_points_mask.reshape(land_mask.shape)

    return land_mask
