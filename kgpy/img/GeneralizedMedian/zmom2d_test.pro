; CCK 2018-Oct-25
; test of zmom2d
Nx = 512
Ny = 256
window,0,xsize=2*Nx,ysize=2*Ny
r = shift( dist(Nx,Ny), Nx/2, Ny/2 )
sig = 10
image = 1/sqrt(1+(r/sig)^2);exp(-r^2/(2*sig^2))

tvscl,image,0
z = zmom2d(image)
tvscl,z,1

image2 = shift(image,150,45) + shift(image,100,-20) + shift(image,-30,60)
tvscl,image2,2
z2 = zmom2d(image2)
tvscl,z2,3

window,1,xsize=2*Nx,ysize=2*Ny
plot_image, image2   
foo = min(z2,sub)
xy = array_indices(z2,sub)               
oplot, [xy[0]],[xy[1]],psym=1,symsize=3

end