/*
 * util.h
 *
 *  Created on: Jun 26, 2018
 *      Author: byrdie
 */

#ifndef SRC_IMG_DSPK_UTIL_H_
#define SRC_IMG_DSPK_UTIL_H_

#include <float.h>

namespace kgpy {

namespace img {

namespace dspk {

class dim3 {
public:
	int x;		// fastest-changing dimension
	int y;
	int z;		// slowest-changing dimension
	int xyz;

	dim3();
	dim3(int x, int y, int z);
	dim3 operator+(dim3 right);
	dim3 operator*(dim3 right);
	dim3 operator/(int right);
};

// define unit vectors along each dimension
const dim3 xhat(1, 0, 0);
const dim3 yhat(0, 1, 0);
const dim3 zhat(0, 0, 1);
const dim3 axes[] = {xhat, yhat, zhat};

class vec3 {
public:
	float x;
	float y;
	float z;

	vec3();
	vec3(float x, float y, float z);
};

class DB {
public:
	dim3 dsz;	// shape of data array
	dim3 ksz;	// shape of local median kernel
	dim3 hsz;	// shape of histograms
	dim3 tsz;	// shape of threshold arrays

	float * data;	// input data array
//	float * lmed;	// local median of data array
	float * gmap;	// map of good pixels
	float * hist;	// histogram of median vs intensity for each axis
	float * cumd;	// cumulative distribution of median vs intensity for each axis
	float * t1;		// upper intensity threshold as a function of median for each axis
	float * t9;		// lower intensity threshold as a function of median for each axis

	float dmax;		// maximum value of the data array
	float dmin;		// minimum value of the data array
	dim3 mmax;		// maximum median value along each axis
	dim3 mmin;		// minimum median value along each axis

	DB(float * data, dim3 dsz, dim3 ksz);
};

float find_max(float * data, dim3 dsz);
float find_min(float * data, dim3 dsz);

}

}

}


#endif /* SRC_IMG_DSPK_UTIL_H_ */
