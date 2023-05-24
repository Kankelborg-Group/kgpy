import dataclasses
import matplotlib.colors
import matplotlib.axes
import matplotlib.animation
import astropy.units as u
import astropy.time
import astropy.coordinates
import numpy as np

from . import sji, spectrograph, mosaics, studies

__all__ = ['mosaics', 'studies']


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
            max_doppler_shift: u.Quantity = 50 * u.km / u.s
    ):

        data_sji = self.sji_images.intensity[:, channel_index]
        data_sg = self.sg_cubes.window_doppler().colors(max_doppler_shift)[:, channel_index]
        data_sg_projected = np.zeros(data_sji.shape + (4,))

        wcs_sg = self.sg_cubes.wcs[:, channel_index]

        for i, wcs_sg_i in enumerate(wcs_sg):

            wcs_sg_i = wcs_sg_i.dropaxis(0)
            time = self.sg_cubes.time[i, channel_index] + self.sg_cubes.exposure_length[i, channel_index]
            time_wcs = self.sji_images.time_wcs[:, channel_index]

            time_sji = self.sji_images.time[:, channel_index]
            exposure_length_sji_max = self.sji_images.exposure_length[:, channel_index].max()
            time_sji = np.append(
                arr=time_sji.value,
                values=(time_sji[~0] + exposure_length_sji_max).value,
            )
            time_sji = astropy.time.Time(time_sji, format='unix')
            indices_time = np.digitize(time.value, bins=time_sji.value)
            indices_time = np.minimum(indices_time, self.sji_images.shape[self.sji_images.axis.time] - 1)

            pix_sg = np.mgrid[:data_sg.shape[~2], 10:data_sg.shape[~1] - 10]
            pix_sg_world = wcs_sg_i.array_index_to_world_values(pix_sg[0], pix_sg[1])

            for j, index_time in enumerate(indices_time):
                wcs_sji_j = self.sji_images.wcs[index_time, channel_index]

                pix_sg_sji_j = wcs_sji_j.world_to_array_index_values(pix_sg_world[1][j], pix_sg_world[0][j], time_wcs[index_time])
                pix_sg_sji_j[0][:] = index_time

                mask = pix_sg_sji_j[1] >= data_sg_projected.shape[1]
                pix_sg_sji_j[1][mask] = data_sg_projected.shape[1] - 1

                # for k in [-1, 0]:
                for k in [0]:
                    data_sg_projected[pix_sg_sji_j[0], pix_sg_sji_j[1], pix_sg_sji_j[2] + k, :] = data_sg[i, j, ..., 10:-10, :]
                    for a in range(len(indices_time)):
                        if (pix_sg_sji_j[0] + a < data_sg_projected.shape[0]).all():
                            data_sg_projected[pix_sg_sji_j[0] + a, pix_sg_sji_j[1], pix_sg_sji_j[2] + k, :] = data_sg[i, j, ..., 10:-10, :]
                            data_sg_projected[pix_sg_sji_j[0] + a, pix_sg_sji_j[1], pix_sg_sji_j[2] + k, ~0] = 0.97 ** a * data_sg[i, j, ..., 10:-10, ~0]

        img_sji = ax.imshow(
            X=data_sji[0].value,
            cmap='gray',
            norm=matplotlib.colors.PowerNorm(
                gamma=0.5,
                vmin=np.nanpercentile(data_sji, thresh_min).value,
                vmax=np.nanpercentile(data_sji, thresh_max).value,
            ),
            origin='lower',
        )

        img_sg = ax.imshow(
            X=data_sg_projected[0],
            origin='lower',
        )

        text_sji = ax.text(
            x=0,
            y=0,
            s=self.sji_images.time[0, channel_index].strftime('%Y-%m-%d %H:%M:%S'),
            ha='left',
            va='top',
            color='white',
        )

        def func(i: int):
            img_sg.set_data(data_sg_projected[i])
            img_sji.set_data(data_sji[i])
            text_sji.set_text(self.sji_images.time[i, channel_index].strftime('%Y-%m-%d %H:%M:%S'))

        return matplotlib.animation.FuncAnimation(
            fig=ax.figure,
            func=func,
            frames=data_sg_projected.shape[0],
            interval=frame_interval.to(u.ms).value,
        )
