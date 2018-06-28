
#include "pyboost.h"

#include "src/img/dspk/dspk.h"

BOOST_PYTHON_MODULE(libkgpy){

	Py_Initialize();
	np::initialize();   // only needed if you use numpy in the interface

	py::def("dspk_ndarr", kgpy::img::dspk::dspk_ndarr);
}
