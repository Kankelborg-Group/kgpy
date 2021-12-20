import pathlib
import typing as typ
import dataclasses
import copy
import shutil
import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy.ndimage
import matplotlib.cm
import matplotlib.colors
import matplotlib.animation
import astropy.time
import astropy.units as u
import astropy.constants
import astropy.wcs
import astropy.io.fits
import astropy.coordinates
import astropy.modeling
import sunpy.physics.differential_rotation
import sunpy.coordinates.frames
import kgpy.obs
import kgpy.moment
import kgpy.mixin
import kgpy.img

__all__ = ['Cube']


@dataclasses.dataclass
class Cube(kgpy.obs.spectral.Cube):
    time_wcs: typ.Optional[u.Quantity] = None

    @classmethod
    def zeros(cls, shape: typ.Sequence[int]) -> 'Cube':
        self = super().zeros(shape=shape)
        sh = shape[:-cls.axis.num_right_dim]
        self.time = astropy.time.Time(np.zeros(sh + (shape[cls.axis.y],)), format='unix')
        self.time_wcs = np.zeros(self.time.shape) * u.s
        return self

    @classmethod
    def from_archive(
            cls,
            archive: pathlib.Path,
            spectral_window: str = 'Si IV 1394',
    ) -> 'Cube':
        extract_dir = archive.parent / pathlib.Path(archive.stem).stem
        if not extract_dir.exists():
            shutil.unpack_archive(filename=archive, extract_dir=extract_dir)
        path_sequence = sorted(extract_dir.rglob('*.fits'))
        return cls.from_path_sequence(path_sequence=path_sequence, spectral_window=spectral_window)

    @classmethod
    def from_path_sequence(
            cls,
            path_sequence: typ.Sequence[pathlib.Path],
            spectral_window: str = 'Si IV 1394',
    ) -> 'Cube':

        hdu_list = astropy.io.fits.open(str(path_sequence[0]))
        hdu_index = 1
        for h in range(len(hdu_list)):
            try:
                if hdu_list[0].header['TDESC' + str(h)] == spectral_window:
                    hdu_index = h
            except KeyError:
                pass
        hdu = hdu_list[hdu_index]

        base_shape = hdu.data.shape
        self = cls.zeros((len(path_sequence), 1,) + base_shape)
        self.channel = self.channel.value << u.AA

        for i, path in enumerate(path_sequence):
            hdu_list = astropy.io.fits.open(str(path))
            hdu = hdu_list[hdu_index]

            d = hdu.data * u.adu

            # self.intensity[i, c] = np.moveaxis(d, 0, ~0)
            self.intensity[i] = d

            self.time_wcs[i] = hdu_list[~1].data[..., 0] * u.s
            self.time[i] = astropy.time.Time(hdu_list[0].header['STARTOBS']) + self.time_wcs[i]
            self.exposure_length[i] = float(hdu_list[0].header['EXPTIME']) * u.s
            self.channel[:] = float(hdu_list[0].header['TWAVE' + str(hdu_index)]) * u.AA

            wcs = astropy.wcs.WCS(hdu.header)
            self.wcs[i] = wcs

        self.intensity[self.intensity == -200 * u.adu] = np.nan

        return self

    @property
    def intensity_despiked(self):
        return kgpy.img.spikes.identify_and_fix(
            data=self.intensity.value,
            axis=(0, ~2, ~1, ~0),
            kernel_size=5,
        )[0] << self.intensity.unit

    def window_doppler(self, shift_doppler: u.Quantity = 300 * u.km / u.s) -> 'Cube':

        wavl_center = self.channel[0]
        wavl_delta = shift_doppler / astropy.constants.c * wavl_center
        wavl_left = wavl_center - wavl_delta
        wavl_right = wavl_center + wavl_delta

        wcs_new = self.wcs.copy()
        for i in range(self.wcs.size):
            index = np.unravel_index(i, self.wcs.shape)
            pix_left = int(self.wcs[index].world_to_pixel_values(wavl_left.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0])
            pix_right = int(self.wcs[index].world_to_pixel_values(wavl_right.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0]) + 1
            if i == 0:
                intensity = np.empty(self.intensity.shape[:~0] + (pix_right - pix_left,))
            intensity[index] = self.intensity[index][..., pix_left:pix_right]
            wcs_new[index] = wcs_new[index][..., pix_left:pix_right]

        other = Cube(
            intensity=intensity,
            # intensity_uncertainty=self.intensity_uncertainty[..., pix_left:pix_right].copy(),
            wcs=wcs_new,
            time=self.time.copy(),
            time_index=self.time_index.copy(),
            channel=self.channel.copy(),
            exposure_length=self.exposure_length.copy(),
            time_wcs=self.time_wcs.copy()
        )

        return other

    @property
    def colors(self):

        intensity = self.intensity

        wcs = self.wcs[0, 0]
        wavl_center = self.channel[0]
        shift_doppler = 50 * u.km / u.s
        wavl_delta = shift_doppler / astropy.constants.c * wavl_center
        wavl_left = wavl_center - wavl_delta
        wavl_right = wavl_center + wavl_delta
        pix_mask_left = int(wcs.world_to_pixel_values(wavl_left.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0])
        pix_mask_right = int(wcs.world_to_pixel_values(wavl_right.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0]) + 1

        intensity_background = np.nanmedian(intensity, axis=2, keepdims=True)
        intensity_background[..., pix_mask_left:pix_mask_right] = np.median(intensity_background, axis=~0, keepdims=True)
        intensity = intensity - intensity_background

        intensity_max = np.nanpercentile(intensity, 95, self.axis.perp_axes(self.axis.w))
        intensity_min = 0
        intensity = (intensity - intensity_min) / (intensity_max - intensity_min)

        colormap = matplotlib.cm.get_cmap('gist_rainbow')
        segment_data = colormap._segmentdata.copy()

        last_segment = ~1
        segment_data['red'] = segment_data['red'][:last_segment].copy()
        segment_data['green'] = segment_data['green'][:last_segment].copy()
        segment_data['blue'] = segment_data['blue'][:last_segment].copy()
        segment_data['alpha'] = segment_data['alpha'][:last_segment].copy()

        segment_data['red'][:, 0] /= segment_data['red'][~0, 0]
        segment_data['green'][:, 0] /= segment_data['green'][~0, 0]
        segment_data['blue'][:, 0] /= segment_data['blue'][~0, 0]
        segment_data['alpha'][:, 0] /= segment_data['alpha'][~0, 0]

        colormap = matplotlib.colors.LinearSegmentedColormap(
            name='spectrum',
            segmentdata=segment_data,
        )
        mappable = matplotlib.cm.ScalarMappable(
            cmap=colormap.reversed(),
        )

        shift_doppler = 100 * u.km / u.s

        wavl_delta = shift_doppler / astropy.constants.c * wavl_center
        wavl_left = wavl_center - wavl_delta
        wavl_right = wavl_center + wavl_delta
        pix_left = int(wcs.world_to_pixel_values(wavl_left.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0])
        pix_right = int(wcs.world_to_pixel_values(wavl_right.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0]) + 1
        index = np.expand_dims(np.arange(pix_left, pix_right), axis=self.axis.perp_axes(self.axis.w))
        color = np.nansum(mappable.to_rgba(index) * intensity[..., pix_left:pix_right, np.newaxis], axis=~1)
        color = color / np.sum(mappable.to_rgba(index), axis=~1)

        # color[..., ~0] = color[..., :~0].sum(~0) / 3
        color[..., ~0] = color[..., :~0].max(~0)

        # color[..., ~0] = np.sqrt(color[..., ~0]).real
        # color[... :~0] = color[..., :~0] / np.nanmax(color[..., :~0], axis=~0, keepdims=True)

        color_max = np.nanmax(color[..., :~0], axis=~0)
        mask = color_max > 0
        color[mask, :~0] = color[mask, :~0] / color_max[mask, np.newaxis]

        return color

    def animate_colors(
            self,
            ax: matplotlib.axes.Axes,
            channel_index: int = 0,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
            frame_interval: u.Quantity = 1 * u.s,
    ) -> matplotlib.animation.FuncAnimation:

        other = self.window_doppler(shift_doppler=300 * u.km / u.s)
        data = other.colors[:, channel_index]

        pad = 300
        pads = [pad, pad]
        data = np.swapaxes(data, 1, 2)
        data = np.pad(data, pad_width=[[0, 0], pads, pads, [0, 0]])

        index_reference = 0
        slice_reference = index_reference, channel_index

        reference_coordinate = astropy.coordinates.SkyCoord(
            Tx=self.wcs[slice_reference].wcs.crval[2] * u.deg,
            Ty=self.wcs[slice_reference].wcs.crval[1] * u.deg,
            obstime=self.time[slice_reference][0],
            observer="earth",
            frame=sunpy.coordinates.frames.Helioprojective
        )

        for i in range(data.shape[0]):
            if i == index_reference:
                continue

            reference_coordinate_rotated = sunpy.physics.differential_rotation.solar_rotate_coordinate(
                coordinate=reference_coordinate,
                time=self.time[i, channel_index][0],
            )

            shift_x = reference_coordinate_rotated.Tx - self.wcs[i, channel_index].wcs.crval[2] * u.deg
            shift_y = reference_coordinate_rotated.Ty - self.wcs[i, channel_index].wcs.crval[1] * u.deg

            shift_x = shift_x / (self.wcs[i, channel_index].wcs.cdelt[2] * u.deg)
            shift_y = shift_y / (self.wcs[i, channel_index].wcs.cdelt[1] * u.deg)
            shift = np.rint(np.array([shift_y, shift_x, 0]))
            print(shift)

            data[i] = np.fft.ifftn(scipy.ndimage.fourier_shift(np.fft.fftn(data[i]), -shift)).real

        data = data[..., pad:-pad, :, :]

        img = ax.imshow(
            X=data[0],
            origin='lower',
        )

        def func(i: int):
            img.set_data(data[i])

        return matplotlib.animation.FuncAnimation(
            fig=ax.figure,
            func=func,
            # frames=20,
            frames=data.shape[0],
            interval=frame_interval.to(u.ms).value,
        )


@dataclasses.dataclass
class CubeList(kgpy.mixin.DataclassList[Cube]):

    def to_cube(self) -> Cube:
        def concat(param: str):
            return np.concatenate([getattr(cube, param) for cube in self])

        return Cube(
            intensity=concat('intensity'),
            # intensity_uncertainty=concat('intensity_uncertainty'),
            wcs=concat('wcs'),
            time=astropy.time.Time(np.concatenate([img.time.value for img in self]), format='unix'),
            time_index=concat('time_index'),
            channel=concat('channel'),
            exposure_length=concat('exposure_length'),
            time_wcs=concat('time_wcs'),
        )
