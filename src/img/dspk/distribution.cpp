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
	vec3 mmax = db->mmax;
	vec3 mmin = db->mmin;

	// split data size into single variables
	int dx = dsz.x;
	int dy = dsz.y;
	int dz = dsz.z;

	// compute array strides in each dimension
	int sx = 1;
	int sy = sx * dx;
	int sz = sy * dy;

	vec3 ax = vec3(axes[axis]);

	// calculate the extreme median values
	float med_max = find_max(lmed, gmap, dsz);
	float med_min = find_min(lmed, gmap, dsz);
	vec3 this_mmax = ax * med_max;

#pragma acc parallel loop collapse(3) present(data), present(gmap)
	for(int z = 0; z < dz; z++){	// loop along z axis of data array
		for(int y = 0; y < dy; y++){	// loop along y axis of of data array
			for(int x = 0; x < dx; x++){	// loop along x axis of of data array

				// overall linear index of data array
				int L =  (sz * z) + (sy * y) + (sx * x);



			}
		}
	}

}
void calc_cumulative_distribution(DB * db, int axis);

}

int hist2data(int hval, float m_min, float m_max, int nbins){

	float delta = (m_max - m_min) / ((float) nbins);

	float val = hval * delta + m_min;

	return val;

}
int data2hist(float dval, float m_min, float m_max, int nbins){

	float delta = (m_max - m_min) / ((float) nbins);

	int index = floor((dval - m_min) / delta);

	return index;

}

}

}


