PRO TEST_DSPK

  testfile = '/home/byrdie/Kankelborg-Group/kgpy/data/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits'
  
  data = float(mrdfits(testfile, 4))
  
  kx = 25
  ky = 25
  kz = 25
  
  thresh_min = 0.02
  thresh_max = 0.98
  
  bad_pix_val = -200
  
  dspk, data, thresh_min, thresh_max, kz, ky, kx, bad_pix_val
  
  xstepper, data

END