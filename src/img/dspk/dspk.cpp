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

void dspk(DB * db, float tmin, float tmax, float bad_pix_val){

	calc_gmap(db, tmin, tmax, bad_pix_val);

	fix_badpix(db);

#pragma acc exit data copyout(db->data[0:db->dsz.xyz])
#pragma acc exit data copyout(db->gmap[0:db->dsz.xyz])
#pragma acc exit data copyout(db->hist[0:db->hsz.xyz])
#pragma acc exit data copyout(db->cumd[0:db->hsz.xyz])
#pragma acc exit data copyout(db->cnts[0:db->tsz.xyz])
#pragma acc exit data copyout(db->ihst[0:db->tsz.x])
#pragma acc exit data copyout(db->icmd[0:db->tsz.x])
#pragma acc exit data copyout(db->t1[0:db->tsz.xyz])
#pragma acc exit data copyout(db->t9[0:db->tsz.xyz])


}

py::tuple dspk_ndarr(np::ndarray & data, float thresh_min, float thresh_max, int kz, int ky, int kx, float bad_pix_val){

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
	dspk(db, thresh_min, thresh_max, bad_pix_val);

	// return median array
	py::object down = py::object();
	py::tuple dshape = py::make_tuple(dsz.z, dsz.y, dsz.x);
	py::tuple dstride = py::make_tuple(sz, sy, sx);
	np::dtype dtype = np::dtype::get_builtin<float>();

	int hx = sizeof(float);
	int hy = hx * db->hsz.x;
	int hz = hy * db->hsz.y;

	int tx = sizeof(float);
	int ty = tx * db->hsz.x;
	int tz = ty * 1;

	py::object hown = py::object();
	py::tuple hshape = py::make_tuple(db->hsz.z, db->hsz.y, db->hsz.x);
	py::tuple hstride = py::make_tuple(hz, hy, hx);

	py::object town = py::object();
	py::tuple tshape = py::make_tuple(db->hsz.z, 1, db->hsz.x);
	py::tuple tstride = py::make_tuple(tz, ty, tx);

	py::object iown = py::object();
	py::tuple ishape = py::make_tuple(db->hsz.x);
	py::tuple istride = py::make_tuple(tx);

	np::ndarray pydata = np::from_data(db->gmap, dtype, dshape, dstride, down);
	np::ndarray pyhist = np::from_data(db->hist, dtype, hshape, hstride, hown);
	np::ndarray pyt1 = np::from_data(db->t1, dtype, tshape, tstride, town);
	np::ndarray pyt9 = np::from_data(db->t9, dtype, tshape, tstride, town);
	np::ndarray pycnts = np::from_data(db->cnts, dtype, tshape, tstride, town);
	np::ndarray pyihst = np::from_data(db->icmd, dtype, ishape, istride, iown);

	return make_tuple(pydata, pyhist, pyt1, pyt9, pycnts, pyihst);

}

}

}

}


