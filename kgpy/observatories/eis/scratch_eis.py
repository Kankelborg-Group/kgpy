import pathlib
import numpy as np
import matplotlib.pyplot as plt
from kgpy.observatories.eis import eis_spectrograph

data_path = pathlib.Path(__file__).parent / 'data'
directory = data_path / 'launch/l1'

frame_paths = np.array(sorted(directory.glob('*')))
sp  = eis_spectrograph.read_eis_spectrograph_level1_fits(frame_paths, uncertainty=False)

spectral_win = list(sp.data.keys())
print(spectral_win)
cube_sequence = sp.data['He II 256.320']
print(cube_sequence[2].wcs)

raster_img = np.transpose(np.sum(cube_sequence[2].data, -1))
print(raster_img.shape)
raster_img[raster_img > 100000] = 0
plt.imshow(raster_img,origin = 'lower', vmax = np.percentile(raster_img,99.9))

# hdu = data.load_hdu(frame_paths, hdu_index=1)
# spectral_win = np.array([hdu[0].data[i][0] for i in range(len(hdu[0].data))])
# print(spectral_win.shape)
#
#
#
# # plt.imshow(hdu[0].data[1][2])
# print(type(hdu[0].data))
#
#
# # print(hdu[0].header['CDELT3'])
# # print(hdu[0].header['CRVAL2'])
# # print(hdu[0].header['CRPIX2'])
# # print(hdu[0].header['CDELT2'])
# # print(hdu[0].header['CTYPE2'])
#
#
# # print(hdu[0].header['NWIN'])
# # print(hdu[0].header['TFORM1'])
# # print(hdu[0].header['TTYPE1'])
# # print(hdu[0].header['TDESC1'])
# # print(hdu[0].header['DATE_END'])
# # print(hdu[0].header['TTYPE6'])
# # print(hdu[0].header['TTYPE7'])
#
#
# print(hdu[0].header)
plt.show()

# eis_launch = eis.EIS.from_path('eis', launch_path)