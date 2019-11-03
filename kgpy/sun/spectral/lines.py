
import astropy.units as u

from kso.spectral import Line

SiIV_1394 = Line(ion='Si IV', intensity=10, center=1393.8 * u.AA, width=0.1058 * u.AA)

HeII_304 = Line(ion='He II', intensity=9510 * u.W / (u.cm ** 2) / u.steradian, center=303.732 * u.AA, width=0.111/2.355 * u.AA)