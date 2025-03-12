import pandas as pd
import pygmt
import pytest
import xarray as xr

from pygmt_helper.plotting import (
    create_grid,
    gen_region_fig,
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
