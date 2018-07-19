/*
 * util.h
 *
 *  Created on: Jun 26, 2018
 *      Author: byrdie
 */

#ifndef SRC_IMG_DSPK_UTIL_H_
#define SRC_IMG_DSPK_UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <string>

#include <float.h>

namespace kgpy {

namespace img {

namespace dspk {

const float good_pix = 1.0f;
const float bad_pix = 0.0f;

class dim3;
class vec3;

class dim3 {
public:
	int x;		// fastest-changing dimension
	int y;
	int z;		// slowest-changing dimension
	int xyz;

	dim3();
	dim3(int x, int y, int z);
	dim3 operator+(dim3 right);
	int operator*(dim3 right);
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
	vec3(dim3 X);
	vec3 operator*(float right);
	vec3 operator+(vec3 right);
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
	float * cnts;	// number of pixels with each median value
	float * ihst;	// 1D histogram of intensity
	float * icmd;	// 1D cumulative distribution of intensity
	float * t1;		// upper intensity threshold as a function of median for each axis
	float * t9;		// lower intensity threshold as a function of median for each axis
	float i1;		// upper hard intensity threshold
	float i9;		// lower hard intensity threshold

	float dmax;		// maximum value of the data array
	float dmin;		// minimum value of the data array
	vec3 mmax;		// maximum median value along each axis
	vec3 mmin;		// minimum median value along each axis

	DB(float * data, dim3 dsz, dim3 ksz);
};

float find_max(float * data, float * gmap, dim3 dsz);
float find_min(float * data, float * gmap, dim3 dsz);

}

}

}


#endif /* SRC_IMG_DSPK_UTIL_H_ */
