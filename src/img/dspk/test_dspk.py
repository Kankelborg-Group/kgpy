
import os
import time
import matplotlib.pyplot as plt
from astropy.io import fits

from src.img.dspk.dspk import dspk_3D
from src.plot.slice.image_stepper import CubeSlicer

print(os.getcwd())

# testfile = "../../../data/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits"
testfile = '/home/byrdie/Kankelborg-Group/kgpy/data/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits'

hdul = fits.open(testfile)

# hdul.info()

# hdu = hdul['Si IV 1403']
# print(repr(hdul[4].header))

hdu = hdul[4]
data = hdu.data

print(data.shape)



start = time.time()
lmed = dspk_3D(data)
end = time.time()
print('time elapsed', end-start)

dplt = CubeSlicer(data)
mplt = CubeSlicer(lmed)

plt.show()