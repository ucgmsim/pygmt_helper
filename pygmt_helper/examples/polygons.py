from pygmt_helper import plotting

import pygmt
import shapely

region = (170, 175, -45, -40)
projection = "M10c"

fig = plotting.gen_region_fig(
    region=region,
    title="Test Map",
    subtitle="Test Subtitle",
    projection=projection,
    config_options={"MAP_FRAME_PEN": "2p"},
)


test_polygon = shapely.Polygon(
          [
            [
              170.50175961398025,
              -43.43397759003748
            ],
            [
              170.03920443841383,
              -43.88211179838194
            ],
            [
              170.61151846919984,
              -44.265124035523286
            ],
            [
              171.81886587660995,
              -43.74633923909617
            ],
            [
              171.00351438069657,
              -43.24605985518982
            ],
            [
              170.50175961398025,
              -43.43397759003748
            ]
        ],

)


plotting.plot_geometry(fig, test_polygon, pen='1p,blue,-', crs='4326')
plotting.label_polygon_on_boundary(fig, test_polygon, 'Outside!', align=True, projection=projection, fill='white', pen='1p,black', offset='0.3c')
plotting.label_geometry_inside(fig, test_polygon, 'Inside!', fill='white', pen='1p,black')
plotting.label_polygon_at(fig, 0.35, test_polygon, 'Around!', align=True, projection=projection, fill='white', pen='1p,black', offset='0.3c', justify='TC')

fig.savefig('polygons.jpeg')
