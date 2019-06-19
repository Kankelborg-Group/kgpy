import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import astropy.units as u
from matplotlib.animation import ArtistAnimation, writers

from irispy.spectrograph import read_iris_spectrograph_level2_fits
from irispy.sji import read_iris_sji_level2_fits

import kgpy


def prep_sji(path, files, raster_len):

    sji_files = [f for f in files if 'SJI' in f]

    sji_file = next((f for f in sji_files if '1400' in f), None)

    if sji_file is None:
        sji_file = sji_files[0]

    sji_obs = read_iris_sji_level2_fits(os.path.join(path, sji_file))

    sji_data = sji_obs.data

    ratio = raster_len / sji_data.shape[0]

    d = np.zeros((raster_len,) + sji_data.shape[1:])

    for i in range(d.shape[0]):

        n = int(np.floor(i / ratio))

        f = sji_data[n, ...]

        g = np.empty_like(f)

        j = ~np.isnan(f)

        if i > 0:

            g = d[i - 1, ...]

        g[j] = f[j]

        d[i, ...] = g

    return d


def prep_raster(path, files):

    raster_files = [f for f in files if 'raster' in f]

    win_str_0 = 'Si IV 1394'
    win_str_1 = 'Si IV 1403'
    raster_files = [os.path.join(path, f) for f in raster_files]
    print(raster_files)
    raster_obs = read_iris_spectrograph_level2_fits(raster_files)

    if win_str_0 in raster_obs.data:
        raster_obs = raster_obs.data[win_str_0]
        _, _, pix_min = raster_obs.data[0].world_to_pixel(0*u.deg, 0*u.deg, 1392 * u.AA)
        _, _, pix_max = raster_obs.data[0].world_to_pixel(0*u.deg, 0*u.deg, 1396 * u.AA)
    else:
        raster_obs = raster_obs.data[win_str_1]
        _, _, pix_min = raster_obs.data[0].world_to_pixel(0*u.deg, 0*u.deg, 1401 * u.AA)
        _, _, pix_max = raster_obs.data[0].world_to_pixel(0*u.deg, 0*u.deg, 1405 * u.AA)

    pix_min = int(pix_min.value)
    pix_max = int(pix_max.value)

    raster_data = np.vstack([c.data for c in raster_obs.data])

    raster_times = np.concatenate([c.extra_coords['time']['value'] for c in raster_obs.data])

    raster_data = raster_data[..., pix_min:pix_max]

    return raster_data, raster_times


def prep_data(path, files):

    print(path)

    raster_data, raster_times = prep_raster(path, files)
    sji_data = prep_sji(path, files, raster_data.shape[0])

    raster_data[raster_data < -100] = 0

    raster_data[np.isnan(raster_data)] = 0
    sji_data[np.isnan(sji_data)] = 0

    sji_data[sji_data < -100] = 0

    raster_data = raster_data / np.percentile(raster_data, 99.5)
    sji_data = sji_data / np.percentile(sji_data, 99.5)

    data = np.concatenate([raster_data, sji_data], axis=-1)

    return data, raster_times


def movie_gen(input_path, output_path):

    # base_path = os.path.join(os.path.dirname(__file__), 'data')
    #
    # movie_path = os.path.join(os.path.dirname(__file__), 'movies')

    print(input_path)

    for path, dirs, files in os.walk(input_path):

        if not files:
            continue

        print(path)

        name = os.path.split(path)[-1]

        data, times = prep_data(path, files)

        data[data < -100] = 0

        vmin = np.percentile(data, 1)
        vmax = np.percentile(data, 99)

        c = kgpy.plot.CubeSlicer(data, vmin=vmin, vmax=vmax)

        print(name)
        plt.show()

        # fig = plt.figure(figsize=[12.8, 9.6])
        # ims = []
        # for i in range(data.shape[0]):
        #     ims.append((plt.imshow(data[i, ...], vmin=0.0, vmax=1.0, cmap='OrRd_r', origin='lower'),
        #                 plt.annotate(name + ' ' + str(times[i]) + ' ' + str(i), (0.0, 0.0))))
        #
        # fps = 25
        # im_ani = ArtistAnimation(fig, ims, 1000/fps, blit=True)
        # Writer = writers['ffmpeg']
        # writer = Writer(fps=fps, bitrate=1000)
        # im_ani.save(os.path.join(output_path, name + '.mp4'), writer=writer)
        #
        # plt.close(fig)


if __name__ == '__main__':

    movie_gen(sys.argv[1], sys.argv[2])
