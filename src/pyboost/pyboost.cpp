
#include "pyboost.h"

BOOST_PYTHON_MODULE(libkgpy){

	Py_Initialize();
	np::initialize();   // only needed if you use numpy in the interface


}
