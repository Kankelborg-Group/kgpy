PRO DSPK, data, thresh_min, thresh_max, kz, ky, kx, bad_pix_val

  dsz = SIZE(data)
  
  dz = dsz[1]
  dy = dsz[2]
  dx = dsz[3]

  R = CALL_EXTERNAL('/home/byrdie/Kankelborg-Group/kgpy/Release/libkgpy.so', 'dspk_idl', data, thresh_min, thresh_max, dz, dy, dx, kx, ky, kx, bad_pix_val)

END