import subprocess

import pandas as pd
import pygmt
import pytest
import xarray as xr

from pygmt_helper.plotting import (
    DEFAULT_PROJECTION,
    ProjectedRegion,
    _gmt_mapproject,
    create_grid,
    gen_region_fig,
    plot_grid,
)


def test_gen_region_fig_custom_region():
    fig = gen_region_fig(
        region=ProjectedRegion.from_box(170, 175, -45, -40, "M10c"),
        title="Test Map",
        subtitle="Test Subtitle",
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


NZ_CORNERS = (166.0020348, -45.8728952, 179.8740906, -38.036145)


def _map_dimensions(region: ProjectedRegion) -> tuple[float, float]:
    """The figure's width and height in cm, as GMT computes them."""
    width, height = subprocess.run(
        ["gmt", "mapproject", f"-R{region}", f"-J{region.projection}", "-W"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return float(width), float(height)


def test_region_from_rotated_corners_orders_corners_for_gmt():
    """`+r` needs lower-left/upper-right *in rotated space*; a lon/lat SW/NE
    pair usually lands on the anti-diagonal there, which makes GMT compute a
    negative map height instead of normalising."""
    region = ProjectedRegion.from_rotated_corners(*NZ_CORNERS, azimuth=34, width="18c")

    width, height = _map_dimensions(region)
    assert width == pytest.approx(18)
    assert height > 0

    # Corner order must not matter (same diagonal, swapped).
    swapped = ProjectedRegion.from_rotated_corners(
        179.8740906, -38.036145, 166.0020348, -45.8728952, azimuth=34, width="18c"
    )
    assert str(swapped) == str(region)

    # The bounding box must cover all four corners, so it is wider in latitude
    # than the two given corners alone.
    _, _, south, north = region.bounding_box
    assert south < -45.8728952
    assert north > -38.036145


def test_region_from_rotated_corners_builds_projection_about_centre():
    """The oblique Mercator projection is centred on the rectangle's
    centre point (the midpoint of the diagonal defined by the corners),
    with the given azimuth and width passed through unchanged.

    Also checks that a shallower azimuth (rectangle closer to the
    oblique equator) produces a smaller map height than a steeper one,
    confirming the height responds to azimuth as expected."""
    region = ProjectedRegion.from_rotated_corners(*NZ_CORNERS, azimuth=34, width="18c")

    assert region.projection.startswith("Oa")
    centre_lon, centre_lat, azimuth, width = region.projection[2:].split("/")
    assert float(centre_lon) == pytest.approx(173.3657, abs=1e-3)
    assert float(centre_lat) == pytest.approx(-42.1652, abs=1e-3)
    assert azimuth == "34"
    assert width == "18c"

    shallower = ProjectedRegion.from_rotated_corners(
        *NZ_CORNERS, azimuth=20, width="18c"
    )
    assert _map_dimensions(shallower)[1] > _map_dimensions(region)[1]


AUCKLAND = (174.7655, -36.8503)
INVERCARGILL = (168.3538, -46.4132)


def test_region_from_rotated_corners_vertical_is_portrait_and_north_up():
    """`vertical` turns the projection 90 degrees rather than using GMT's
    `+v`, which renders this region south-up with no way to flip it."""
    landscape = ProjectedRegion.from_rotated_corners(
        *NZ_CORNERS, azimuth=34, width="6c"
    )
    portrait = ProjectedRegion.from_rotated_corners(
        *NZ_CORNERS, azimuth=34, width="6c", vertical=True
    )

    assert "+v" not in portrait.projection
    assert portrait.projection.split("/")[2] == "124"

    # Portrait: taller than wide, and the reverse of the landscape aspect.
    p_width, p_height = _map_dimensions(portrait)
    l_width, l_height = _map_dimensions(landscape)
    assert p_height > p_width
    assert p_height / p_width == pytest.approx(l_width / l_height, rel=0.01)

    # North must be up - Auckland above Invercargill on the plotted page.
    (_, auckland_y), (_, invercargill_y) = _gmt_mapproject(
        [AUCKLAND, INVERCARGILL], str(portrait), portrait.projection
    )
    assert auckland_y > invercargill_y


def test_region_from_box():
    region = ProjectedRegion.from_box(166.3, 178.65, -47.05, -35.5, "M8.5c")

    assert str(region) == "166.3/178.65/-47.05/-35.5"
    assert region.bounding_box == (166.3, 178.65, -47.05, -35.5)
    assert region.projection == "M8.5c"
    assert region.corners is None


def test_region_from_gmt_region():
    region = ProjectedRegion.from_gmt_region("NZ")

    assert str(region) == "NZ"
    assert region.projection == DEFAULT_PROJECTION
    # A named region is opaque, so no bounds can be derived from it.
    assert region.bounding_box is None
