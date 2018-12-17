;+
;NAME:
;  ZMOM2D
;PURPOSE:
;  Calculate the zeroth moment sum about every point in a 2D image.
;  The definition is:
;     z(r0) = \sum |r-r0|I(r)
;  where the sum is over all pixel positions r, I(r) is the intensity, 
;  and r0 is fixed position.
;CALLING SEQUENCE:
;  z = zmom2d(I)
;INPUTS:
;  I = a 2D floating point image
;OUTPUT:
;  z = an image, the same size as I, containing the result of the zeroth
;     moment sum, as defined above.
;MODIFICATION HISTORY:
;  2018-Oct-25 C. Kankelborg
function zmom2d, image
;-

isize = size(image)
Nx = isize[1]
Ny = isize[2]
z = fltarr(Nx,Ny)
x = findgen(Nx) - Nx/2
y = findgen(Ny) - Ny/2
xa = x # replicate(1.0,Ny)
ya = replicate(1.0,Nx) # y
r = sqrt(xa^2 + ya^2)

for i = 0,Nx-1 do begin
   for j = 0,Ny-1 do begin
      dx = x[i]
      dy = y[j]
      z[i,j] = total( shift(image,-dx,-dy) * r )
   endfor
endfor

return, z

end