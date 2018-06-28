/*
 * dspk.cpp
 *
 *  Created on: Jun 26, 2018
 *      Author: byrdie
 */

#include "dspk.h"

using namespace std;

namespace kgpy {

namespace img {

namespace dspk {

void dspk(DB * db, float tmin, float tmax){



	float * data = db->data;
	float * lmed = db->lmed;
	float * gmap = db->gmap;
	dim3 dsz = db->dsz;
	dim3 ksz = db->ksz;

	dim3 ax = zhat;



	calc_local_median(lmed, data, gmap, dsz, ksz, ax);


}

np::ndarray dspk_ndarr(np::ndarray & data, float thresh_min, float thresh_max, int kz, int ky, int kx){

	// save size of data array
	const Py_intptr_t * dsh = data.get_shape();
	dim3 dsz(dsh[2], dsh[1], dsh[0]);

	// calculate stride of data array
	int sx = sizeof(float);
	int sy = sx * dsz.x;
	int sz = sy * dsz.y;

	// save size of kernel
	dim3 ksz(kx, ky, kz);

	// extract pointer to data array
	float * dat = (float * ) data.get_data();

	// construct dspk database object
	DB * db = new DB(dat, dsz, ksz);

	// call dspk routine
	dspk(db, thresh_min, thresh_max);

	// return median array
	py::object own = py::object();
	py::tuple shape = py::make_tuple(dsz.z, dsz.y, dsz.x);
	py::tuple stride = py::make_tuple(sz, sy, sx);
	np::dtype dtype = np::dtype::get_builtin<float>();

	return np::from_data(db->lmed, dtype, shape, stride, own);

}

}

}

}


