import kgpy.sun
import kgpy.chianti


def test_spectrum_qs_tr():
    bunch = kgpy.sun.spectrum_qs_tr()
    assert isinstance(bunch, kgpy.chianti.Bunch)