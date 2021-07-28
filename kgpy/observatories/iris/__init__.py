import dataclasses
import matplotlib.axes
import matplotlib.animation
import astropy.units as u
import numpy as np

from . import sji, spectrograph, mosaics

__all__ = ['mosaics']


@dataclasses.dataclass
class Obs:
    sji_images: sji.Image
    sg_cubes: spectrograph.Cube

    def animate_channel_color(
            self,
            ax: matplotlib.axes.Axes,
            channel_index: int = 0,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
            frame_interval: u.Quantity = 1 * u.s,
    ):

        data_sji = self.sji_images.intensity[:, channel_index]
        data_sg = self.sg_cubes.window_doppler().colors[:, channel_index]
        data_sg_projected = np.zeros(data_sg.shape[:2] + data_sji.shape[~1:] + (4,))
        data_sg_projected = np.zeros(data_sji.shape + (4,))

        wcs_sji = self.sji_images.wcs[0, channel_index]
        wcs_sg = self.sg_cubes.wcs[:, channel_index]

        for i, wcs_sg_i in enumerate(wcs_sg):

            wcs_sg_i = wcs_sg_i.dropaxis(0)
            time = (self.sg_cubes.time[i] - self.sji_images.time.min()).to(u.s)

            pix_sg = np.mgrid[:data_sg.shape[~2], 10:data_sg.shape[~1]-10]
            pix_sg_world = wcs_sg_i.array_index_to_world_values(pix_sg[0], pix_sg[1])

            indices_time = wcs_sji.world_to_array_index_values(0, 0, time)[0]
            for j, wcs_sji_j in enumerate(self.sji_images.wcs[indices_time, channel_index]):

                pix_sg_sji_j = wcs_sji_j.world_to_array_index_values(pix_sg_world[1][j], pix_sg_world[0][j], time[j])

                # for k in [-1, 0, 1]:
                for k in [0]:
                    data_sg_projected[pix_sg_sji_j[0], pix_sg_sji_j[1], pix_sg_sji_j[2] + k, :~0] = data_sg[i, j, ..., 10:-10, :]
                    data_sg_projected[pix_sg_sji_j[0], pix_sg_sji_j[1], pix_sg_sji_j[2] + k, ~0] = 1
                    for a in range(len(indices_time)):
                        if (pix_sg_sji_j[0] + a < data_sg_projected.shape[0]).all():
                            data_sg_projected[pix_sg_sji_j[0] + a, pix_sg_sji_j[1], pix_sg_sji_j[2] + k, :~0] = data_sg[i, j, ..., 10:-10, :]
                            data_sg_projected[pix_sg_sji_j[0] + a, pix_sg_sji_j[1], pix_sg_sji_j[2] + k, ~0] = 0.9 ** a



        img_sji = ax.imshow(
            X=data_sji[0].value,
            cmap='gray',
            vmin=np.nanpercentile(data_sji, thresh_min).value,
            vmax=np.nanpercentile(data_sji, thresh_max).value,
        )

        img_sg = ax.imshow(
            X=data_sg_projected[0],
        )

        def func(i: int):
            img_sg.set_data(data_sg_projected[i])
            img_sji.set_data(data_sji[i])

        return matplotlib.animation.FuncAnimation(
            fig=ax.figure,
            func=func,
            # frames=200,
            frames=data_sg_projected.shape[0],
            interval=frame_interval.to(u.ms).value,
        )



