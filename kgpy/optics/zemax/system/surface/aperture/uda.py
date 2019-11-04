import pathlib
import typing as tp

from astropy import units as u


def write_file(uda_file: pathlib.Path, polygons: tp.List[u.Quantity], zemax_units: u.Unit):
    with open(str(uda_file), 'w') as uda:

        for poly in polygons:

            x_pts = poly[..., 0]
            y_pts = poly[..., 1]

            for x, y in zip(x_pts.flat, y_pts.flat):
                line = 'LIN ' + str(x.to(zemax_units).value) + ' ' + str(y.to(zemax_units).value) + '\n'
                uda.write(line)

            uda.write('BRK\n')