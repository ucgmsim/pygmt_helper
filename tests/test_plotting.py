import geopandas
import numpy as np
import pandas as pd
import pygmt
import pytest
import shapely
import xarray as xr

from pygmt_helper.plotting import (
    NZMapData,
    clip,
    create_grid,
    gen_region_fig,
    get_coast_water_mask,
    plot_grid,
)


def test_gen_region_fig_custom_region():
    fig = gen_region_fig(
        region=(170, 175, -45, -40),
        title="Test Map",
        subtitle="Test Subtitle",
        projection="M10c",
        config_options={"MAP_FRAME_PEN": "2p"},
    )
    assert isinstance(fig, pygmt.Figure)


@pytest.fixture
def data_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lon": [
                172.6110032326659,
                172.6110032326659,
                172.65207064735455,
                172.65207064735455,
            ],
            "lat": [
                -43.51895856877526,
                -43.54083157266359,
                -43.54083157266359,
                -43.51895856877526,
            ],
            "data": [1, 2, 3, 10],
        }
    )


def test_create_grid(data_df: pd.DataFrame):
    grid = create_grid(
        data_df,
        "data",
        grid_spacing="10000e/10000e",
    )
    assert isinstance(grid, xr.DataArray)
    assert grid.shape == (155, 103)


def test_create_grid_cloughtorcher(data_df: pd.DataFrame):
    grid = create_grid(
        data_df,
        "data",
        interp_method="CloughTorcher",
        grid_spacing="10000e/10000e",
    )
    assert isinstance(grid, xr.DataArray)
    assert grid.shape == (155, 103)


def test_create_grid_nearest(data_df: pd.DataFrame):
    grid = create_grid(
        data_df,
        "data",
        interp_method="nearest",
        grid_spacing="10000e/10000e",
    )
    assert isinstance(grid, xr.DataArray)
    assert grid.shape == (155, 103)


def test_create_grid_invalid_interp(data_df: pd.DataFrame):
    with pytest.raises(ValueError):
        create_grid(
            data_df,
            "data",
            interp_method="invalid",
            grid_spacing="10000e/10000e",
        )


def test_plot_grid(data_df: pd.DataFrame):
    fig = pygmt.Figure()
    fig.basemap(region="NZ", projection="M10c", frame=True)
    grid = create_grid(
        data_df,
        "data",
        grid_spacing="10000e/10000e",
    )
    plot_grid(
        fig,
        grid,
        "rainbow",
        (1, 10, 1),
        ("red", "blue"),
        "Test Data",
        plot_contours=False,
    )
    assert isinstance(fig, pygmt.Figure)


@pytest.fixture
def fake_map_data() -> NZMapData:
    land_ring = shapely.LinearRing(
        [(172, -44), (173, -44), (173, -43), (172, -43), (172, -44)]
    )
    lake_ring = shapely.LinearRing(
        [(172.4, -43.6), (172.6, -43.6), (172.6, -43.4), (172.4, -43.4), (172.4, -43.6)]
    )
    return NZMapData(
        road_df=pd.DataFrame(),
        highway_df=geopandas.GeoDataFrame(),
        coastline_df=geopandas.GeoDataFrame(geometry=[land_ring], crs="EPSG:4326"),
        water_df=geopandas.GeoDataFrame(geometry=[lake_ring], crs="EPSG:4326"),
        topo_grid=xr.DataArray(),
        topo_shading_grid=xr.DataArray(),
    )


def test_get_coast_water_mask(fake_map_data: NZMapData):
    points = np.array(
        [
            [-43.5, 171.5],  # offshore
            [-43.8, 172.2],  # on land, not in lake
            [-43.5, 172.5],  # in the lake
        ]
    )
    coast_mask, water_mask = get_coast_water_mask(fake_map_data, points)
    np.testing.assert_array_equal(coast_mask, [False, True, True])
    np.testing.assert_array_equal(water_mask, [False, False, True])


def test_clip():
    """Test that the clip context manager doesn't crash with a simple polygon."""
    # Create a simple rectangular polygon
    polygon = shapely.Polygon([(170, -45), (175, -45), (175, -40), (170, -40)])

    # Test that the context manager can be entered and exited without errors
    with clip([polygon]):
        # If we reach here without exception, the basic functionality works
        pass
