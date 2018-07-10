/*
 * distribution.cpp
 *
 *  Created on: Jun 28, 2018
 *      Author: byrdie
 */

#include "distribution.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_histogram(DB * db, float * lmed, int axis){

	// extract requisite data from database
	float * data = db->data;
	float * gmap = db->gmap;
	float * hist = db->hist;
	dim3 dsz = db->dsz;
	dim3 hsz = db->hsz;
	float dmax = db->dmax;
	float dmin = db->dmin;

	// split data size into single variables
	int dx = dsz.x;
	int dy = dsz.y;
	int dz = dsz.z;

	// compute data strides in each dimension
	int sx = 1;
	int sy = sx * dx;
	int sz = sy * dy;

	// compute histogram strides
	int hx = 1;
	int hy = hx * hsz.x;
	int hz = hy * hsz.y;

//	vec3 ax = vec3(axes[axis]);
//
//	// calculate the extreme median values
//	float med_max = find_max(lmed, gmap, dsz);
//	float med_min = find_min(lmed, gmap, dsz);
//
//	// save extreme median value to database for this axis
//	vec3 mmax = ax * med_max;
//	vec3 mmin = ax * med_min;
//	db->mmax = db->mmax + mmax;
//	db->mmin = db->mmin + mmin;


//	printf("%e %e %e %e", dmin, dmax, med_min, med_max);

#pragma acc parallel loop collapse(3) present(data), present(gmap)
	for(int z = 0; z < dz; z++){	// loop along z axis of data array
		for(int y = 0; y < dy; y++){	// loop along y axis of of data array
			for(int x = 0; x < dx; x++){	// loop along x axis of of data array

				// overall linear index of data array
				int L =  (sz * z) + (sy * y) + (sx * x);



				// Don't incorporate pixels already marked as bad
				if (gmap[L] == bad_pix) {
					continue;
				}

				// load histogram values
				float d = data[L];
				float m = lmed[L];

				// calculate histogram indices
				int X = data2hist(m, dmin, dmax, hsz.x);
				int Y = data2hist(d, dmin, dmax, hsz.y);

//				printf("%f %f %d %d\n", m, d, X, Y);

				// update histogram
				int H = hz * axis + hy * Y + hx * X;
#pragma acc atomic update
				hist[H] = hist[H] + 1.0f;

			}
		}
	}

}
void calc_cumulative_distribution(DB * db, int axis){

	// extract requisite data from database
	float * hist = db->hist;
	float * cumd = db->cumd;
	float * cnts = db->cnts;
	dim3 hsz = db->hsz;

	// compute histogram strides
	int hx = 1;
	int hy = hx * hsz.x;
	int hz = hy * hsz.y;

#pragma acc parallel loop present(hist), present(cumd), present(cnts)
	for(int x = 0; x < hsz.x; x++){

		float sum = 0.0f;

		// march along y to build cumulative distributi
		for(int y = 0; y < hsz.y; y++){

			// linear index in histogram
			int H = hz * axis + hy * y + hx * x;

			// increment sum
			sum = sum + hist[H];

			// store result
			cumd[H] = sum;

		}

		// save number of counts for each median
		int C = x + hsz.x * axis;
		cnts[C] = sum;

		// normalize
		for(int y = 0; y < hsz.y; y++){

			// linear index in histogram
			int H = hz * axis + hy * y + hx * x;

			// normalize cumulative distribution
			if (sum != 0.0f) {
				cumd[H] = cumd[H] / sum;
			} else {
				cumd[H] = 0.0f;
			}

			// normalize histogram
			if (sum != 0.0f) {
				hist[H] = hist[H] / sum;
			} else {
				hist[H] = 0.0f;
			}

		}
	}

}

void init_histogram(DB * db){

	float * hist = db->hist;
	dim3 hsz = db->hsz;

	// compute strides for histogram
	int hx = 1;
	int hy = hx * hsz.x;
	int hz = hy * hsz.y;

#pragma acc parallel loop collapse(3) present(hist)
	for(int z = 0; z < hsz.z; z++){
		for(int y = 0; y < hsz.y; y++){
			for(int x = 0; x < hsz.x; x++){

				int L = x * hx + y * hy + z * hz;

				hist[L] = 0.0f;

			}
		}
	}


}

float hist2data(int hval, float m_min, float m_max, int nbins){

	float delta = (m_max - m_min) / (((float) nbins) - 1.0f);

	float val = hval * delta + m_min;

	return val;

}
int data2hist(float dval, float m_min, float m_max, int nbins){

	float delta = (m_max - m_min) / (((float) nbins) - 1.0f);

	int index = floor((dval - m_min) / delta);

	return index;

}

}

}

}


