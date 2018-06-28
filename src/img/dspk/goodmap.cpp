/*
 * goodmap.cpp
 *
 *  Created on: Jun 28, 2018
 *      Author: byrdie
 */

#include "goodmap.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_gmap(DB * db, float tmin, float tmax, float bad_pix_val){

	float * data = db->data;
	float * gmap = db->gmap;
	dim3 dsz = db->dsz;

	init_gmap(gmap, data, dsz, bad_pix_val);

	int naxis = 3;
	for(int axis = 0; axis < naxis; axis++){

		calc_axis_gmap(db, tmin, tmax, axis);

	}

	return;
}
void calc_axis_gmap(DB * db, float tmin, float tmax, int axis){

	float * data = db->data;
	float * gmap = db->gmap;
	float * hist = db->hist;
	float * t1 = db->t1;
	float * t9 = db->t9;
	dim3 dsz = db->dsz;
	dim3 ksz = db->ksz;

	// allocate array for local median calculation
	float * lmed = new float[dsz.xyz];
#pragma acc data create(lmed[0:dsz.xyz])
	{
		calc_local_median(lmed, data, gmap, dsz, ksz, axis);
//		calc_thresh(t1, hist);
	}

	return;

}

void init_gmap(float * gmap, float * data, dim3 dsz, float bad_pix_val) {

	// split data size into single variables
	int dx = dsz.x;
	int dy = dsz.y;
	int dz = dsz.z;

	// compute array strides in each dimension
	int sx = 1;
	int sy = sx * dx;
	int sz = sy * dy;

#pragma acc parallel loop collapse(3) present(data), present(gmap)
	for(int z = 0; z < dz; z++){	// loop along z axis of data array
		for(int y = 0; y < dy; y++){	// loop along y axis of of data array
			for(int x = 0; x < dx; x++){	// loop along x axis of of data array

				// overall linear index of data array
				int L =  (sz * z) + (sy * y) + (sx * x);

				float p = data[L];	// data value at this coordinate
				float m = 1.0f;		// default goodmap value (assume pixel is good)

				// check if pixel is any of the known bad values
				if(p == NAN){
					m = 0.0f;
				} else if (p == bad_pix_val) {
					m = 0.0f;
				}

				gmap[L] = m;

			}
		}
	}

}

}

}

}

