
#include <kgpy/img/dspk/dspk.h>
#include <kgpy/pyboost/pyboost.h>

BOOST_PYTHON_MODULE(libkgpy){

	Py_Initialize();
	np::initialize();   // only needed if you use numpy in the interface

	py::def("dspk_ndarr", kgpy::img::dspk::dspk_ndarr);

}
