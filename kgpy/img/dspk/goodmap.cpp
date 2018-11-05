/*
 * goodmap.cpp
 *
 *  Created on: Jun 28, 2018
 *      Author: byrdie
 */

#include <kgpy/img/dspk/goodmap.h>

namespace kgpy {

namespace img {

namespace dspk {

void calc_gmap(DB * db, float tmin, float tmax, float bad_pix_val){

	float * data = db->data;
	float * gmap = db->gmap;
	dim3 dsz = db->dsz;

//	printf("Checkpoint 2\n");

	// initialize arrays
	init_gmap(gmap, data, dsz, bad_pix_val);
	init_histogram(db);

//	printf("Checkpoint 3\n");

	// find the extreme values of the data array not masked by the goodmap
	db->dmax = find_max(data, gmap, dsz);
	db->dmin = find_min(data, gmap, dsz);

//	printf("Checkpoint 4\n");

	int naxis = 3;
	for(int axis = 0; axis < naxis; axis++){

		calc_axis_gmap(db, tmin, tmax, axis);

//		printf("Checkpoint 4.%d\n", axis);

	}

//	printf("Checkpoint 5\n");

	finalize_gmap(db);

	return;
}
void calc_axis_gmap(DB * db, float tmin, float tmax, int axis){

	float * data = db->data;
	float * gmap = db->gmap;
	dim3 dsz = db->dsz;
	dim3 ksz = db->ksz;

	// allocate array for local median calculation
	float * lmed = new float[dsz.xyz];
#pragma acc data create(lmed[0:dsz.xyz])
	{

//		printf("Checkpoint 4.%d.1\n", axis);
		calc_local_median(lmed, data, gmap, dsz, ksz, axis);
//		printf("Checkpoint 4.%d\n", axis);
		calc_histogram(db, lmed, axis);
//		printf("Checkpoint 4.%d.2\n", axis);
		calc_cumulative_distribution(db, axis);
//		printf("Checkpoint 4.%d.3\n", axis);
		calc_thresh(db, tmin, tmax, axis);
//		printf("Checkpoint 4.%d.4\n", axis);
		increment_gmap(db, lmed, axis);
	}

	return;

}

void finalize_gmap(DB * db){

	float * gmap = db->gmap;
	dim3 dsz = db->dsz;

	// split data size into single variables
	int dx = dsz.x;
	int dy = dsz.y;
	int dz = dsz.z;

	// compute array strides in each dimension
	int sx = 1;
	int sy = sx * dx;
	int sz = sy * dy;



#pragma acc parallel loop collapse(3) present(gmap)
	for(int z = 0; z < dz; z++){	// loop along z axis of data/median array)
		for(int y = 0; y < dy; y++){	// loop along y axis of of data/median array
			for(int x = 0; x < dx; x++){	// loop along x axis of of data/median array

				// overall linear index of data/median array
				int L =  (sz * z) + (sy * y) + (sx * x);

				float g = gmap[L];

				if (g >= 4.0f) {
					gmap[L] = bad_pix;
				} else if (g > 1.0f) {
					gmap[L] = good_pix;
				}

			}
		}
	}

}

void increment_gmap(DB * db, float * lmed, int axis){

	float * data = db->data;
	float * gmap = db->gmap;
	float * t1 = db->t1;
	float * t9 = db->t9;
	dim3 dsz = db->dsz;
	dim3 hsz = db->hsz;
	float dmax = db->dmax;
	float dmin = db->dmin;

	// split data size into single variables
	int dx = dsz.x;
	int dy = dsz.y;
	int dz = dsz.z;

	// compute array strides in each dimension
	int sx = 1;
	int sy = sx * dx;
	int sz = sy * dy;

	// compute threshold strides
	int tx = 1;
	int tz = tx * hsz.x;


#pragma acc parallel loop collapse(3) present(data), present(gmap), present(lmed)
	for(int z = 0; z < dz; z++){	// loop along z axis of data/median array)
		for(int y = 0; y < dy; y++){	// loop along y axis of of data/median array
			for(int x = 0; x < dx; x++){	// loop along x axis of of data/median array

				// overall linear index of data/median array
				int L =  (sz * z) + (sy * y) + (sx * x);

				// do nothing if bad pixel has already been identified
				if(gmap[L] == bad_pix) continue;

				// load data and median for this pixel
				float d = data[L];
				float m = lmed[L];

				// calculate the position of this median/data pair on the histogram
				int X = d2h(m, dmin, dmax, hsz.x);
				int Y = d2h(d, dmin, dmax, hsz.y);

				// determine if the threshold is exceeded
				int T = axis * tz + X * tx;
				int Y1 = t1[T];
				int Y9 = t9[T];

				if ((Y > Y1) or (Y < Y9)) {
					gmap[L] = gmap[L] + 1.0f;
				}



			}
		}
	}

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
				float m = good_pix;		// default goodmap value (assume pixel is good)

				// check if pixel is any of the known bad values
				if(p == NAN){
					m = bad_pix;
				} else if (p == bad_pix_val) {
					m = bad_pix;
				}

				gmap[L] = m;

			}
		}
	}

}

int d2h(float dval, float m_min, float m_max, int nbins){

	float delta = (m_max - m_min) / (((float) nbins) - 1.0f);

	int index = floor((dval - m_min) / delta);

	return index;

}

}

}

}

