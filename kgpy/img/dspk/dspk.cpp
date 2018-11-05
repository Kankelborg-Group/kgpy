/*
 * dspk.cpp
 *
 *  Created on: Jun 26, 2018
 *      Author: byrdie
 */

#include <kgpy/img/dspk/dspk.h>

using namespace std;

namespace kgpy {

namespace img {

namespace dspk {

void dspk(DB * db, float tmin, float tmax, float bad_pix_val){

	dim3 dsz = db->dsz;

	// split data size into single variables
	int dx = dsz.x;
	int dy = dsz.y;
	int dz = dsz.z;

	// compute array strides in each dimension
	int sx = 1;
	int sy = sx * dx;
	int sz = sy * dy;


	float datamem = ((float)dsz.xyz) * ((float) sizeof(float));
//	printf("datamem = %f\n", datamem);
	if (datamem > 0.5 * 1e9) {
		int pivot = dz / 2;

		float * ldata = db->data;
		float * rdata = db->data + pivot * sz;

		float * lgmap = db->gmap;
		float * rgmap = db->gmap + pivot * sz;

		dim3 ldsz = dim3(dx, dy, pivot);
		dim3 rdsz = dim3(dx, dy, dz - pivot);

		DB * ldb = new DB(ldata, lgmap, ldsz, db->ksz);
		DB * rdb = new DB(rdata, rgmap, rdsz, db->ksz);

		dspk(ldb, tmin, tmax, bad_pix_val);
		dspk(rdb, tmin, tmax, bad_pix_val);
	} else {

#pragma acc enter data copyin(db->data[0:db->dsz.xyz])
#pragma acc enter data create(db->gmap[0:db->dsz.xyz])
#pragma acc enter data create(db->hist[0:db->hsz.xyz])
#pragma acc enter data create(db->cumd[0:db->hsz.xyz])
#pragma acc enter data create(db->cnts[0:db->tsz.xyz])
#pragma acc enter data create(db->ihst[0:db->tsz.x])
#pragma acc enter data create(db->icmd[0:db->tsz.x])
#pragma acc enter data create(db->t1[0:db->tsz.xyz])
#pragma acc enter data create(db->t9[0:db->tsz.xyz])

//	printf("Checkpoint 1\n");
	calc_gmap(db, tmin, tmax, bad_pix_val);

//	printf("Checkpoint 100\n");

	fix_badpix(db);

//	printf("\n");

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
	float * gmap = new float[dsz.xyz];

	// construct dspk database object
	DB * db = new DB(dat, gmap, dsz, ksz);

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
//	np::ndarray pyihst = np::from_data(db->icmd, dtype, ishape, istride, iown);

	return make_tuple(pydata, pyhist, pyt1, pyt9, pycnts);

}

//void dspk_idl(float * data, float thresh_min, float thresh_max, int dz, int dy, int dx,int kz, int ky, int kx, float bad_pix_val){
//
//	dim3 dsz = dim3(dx, dy, dz);
//	dim3 ksz = dim3(kx, ky, kz);
//
//	// construct dspk database object
//	DB * db = new DB(data, dsz, ksz);
//
//	dspk(db, thresh_min, thresh_max, bad_pix_val);
//
//}

}

}

}


