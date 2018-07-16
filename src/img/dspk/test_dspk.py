
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from src.img.dspk.dspk import dspk_3D
from src.plot.slice.image_stepper import CubeSlicer
from matplotlib import colors

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
results = dspk_3D(data, upper_thresh=0.99)
end = time.time()
print('time elapsed', end-start)


gmap = results[0]
histogram = results[1]
t1 = results[2]
t9 = results[3]
cnts = results[4]

dplt = CubeSlicer(data)
mplt = CubeSlicer(gmap)
hplt = CubeSlicer(histogram, norm=colors.SymLogNorm(1e-4), origin='lower')

for ind in range(3):
    plt.figure()
    plt.imshow(histogram[ind,:,:], norm=colors.SymLogNorm(1e-4), origin='lower')
    plt.plot(t1[ind,0,:])
    plt.plot(t9[ind,0,:])

plt.figure()
plt.semilogy(cnts[ind,0,:])

plt.show()