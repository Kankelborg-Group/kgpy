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
	float * hist = db->data;
	dim3 dsz = db->dsz;
	dim3 hsz = db->hsz;
	float dmax = db->dmax;
	float dmin = db->dmin;
	dim3 mmax = db->mmax;
	dim3 mmin = db->mmin;


	// calculate the extreme median values
	float med_max = find_max(lmed, dsz);
	float med_min = find_min(lmed, dsz);



}
void calc_cumulative_distribution(DB * db, int axis);

}

}

}


