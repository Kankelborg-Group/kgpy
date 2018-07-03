/*
 * util.cpp
 *
 *  Created on: Jun 26, 2018
 *      Author: byrdie
 */

#include "util.h"

namespace kgpy {

namespace img {

namespace dspk {

dim3::dim3() : x(0), y(0), z(0) {
	xyz = 1;
}

dim3::dim3(int x, int y, int z) : x(x), y(y), z(z) {

	xyz = x * y * z;

}

dim3 dim3::operator+(dim3 right){

	int X = x + right.x;
	int Y = y + right.y;
	int Z = z + right.z;

	return dim3(X, Y, Z);

}

/**
 * Dot product
 */
int dim3::operator*(dim3 right){

	int X = x * right.x;
	int Y = y * right.y;
	int Z = z * right.z;

	return X + Y + Z;
}

dim3 dim3::operator/(int right){

	int X = x / right;
	int Y = y / right;
	int Z = z / right;

	return dim3(X, Y, Z);

}

vec3::vec3() : x(0.0f), y(0.0f), z(0.0f){

}

vec3::vec3(float x, float y, float z) : x(x), y(y), z(z) {

}

vec3::vec3(dim3 X){
	x = X.x;
	y = X.y;
	z = X.z;
}

vec3 vec3::operator*(float right){

	float X = right * x;
	float Y = right * y;
	float Z = right * z;

	return vec3(X, Y, Z);

}

DB::DB(float * data, dim3 dsz, dim3 ksz) : data(data), dsz(dsz), ksz(ksz){


	int hx = 1024;
	int hy = 1024;
	int hz = 3;
	hsz = dim3(hx, hy, hz);
	tsz = dim3(hx, 1, hz);

//	lmed = new float[dsz.xyz];	// allocate local median array
	gmap = new float[dsz.xyz];	// allocate good pixel map array
	hist = new float[hsz.xyz];
	cumd = new float[hsz.xyz];
	t1 = new float[hsz.x * hsz.z];
	t9 = new float[hsz.x * hsz.z];

#pragma acc enter data copyin(data[0:dsz.xyz])
//#pragma acc enter data create(lmed[0:dsz.xyz])
#pragma acc enter data create(gmap[0:dsz.xyz])
#pragma acc enter data create(hist[0:hsz.xyz])
#pragma acc enter data create(cumd[0:hsz.xyz])
#pragma acc enter data create(t1[0:tsz.xyz])
#pragma acc enter data create(t9[0:tsz.xyz])

	dmax = 0.0f;
	dmin = 0.0f;

}

float find_max(float * data, float * gmap, dim3 dsz){

	// split data size into single variables
	int dx = dsz.x;
	int dy = dsz.y;
	int dz = dsz.z;

	// compute array strides in each dimension
	int sx = 1;
	int sy = sx * dx;
	int sz = sy * dy;

	float maxfield = -FLT_MAX;

#pragma acc parallel loop collapse(3) reduction(max:maxfield) present(data)
	for(int z = 0; z < dz; z++){	// loop along z axis of data array
		for(int y = 0; y < dy; y++){	// loop along y axis of of data array
			for(int x = 0; x < dx; x++){	// loop along x axis of of data array

				// overall linear index of data array
				int L =  (sz * z) + (sy * y) + (sx * x);

				if (gmap[L] == good_pix) {
					float p = data[L];	// data value at this coordinate
					if(p > maxfield){
						maxfield = p;
					}
				}

			}
		}
	}

	return maxfield;

}

float find_min(float * data, float * gmap, dim3 dsz){

	// split data size into single variables
	int dx = dsz.x;
	int dy = dsz.y;
	int dz = dsz.z;

	// compute array strides in each dimension
	int sx = 1;
	int sy = sx * dx;
	int sz = sy * dy;

	float minfield = FLT_MAX;

	#pragma acc parallel loop collapse(3) reduction(min:minfield) present(data)
	for(int z = 0; z < dz; z++){	// loop along z axis of data array
		for(int y = 0; y < dy; y++){	// loop along y axis of of data array
			for(int x = 0; x < dx; x++){	// loop along x axis of of data array

				// overall linear index of data array
				int L =  (sz * z) + (sy * y) + (sx * x);
				if (gmap[L] == good_pix) {
					float p = data[L];	// data value at this coordinate
					if(p < minfield){
						minfield = p;
					}
				}

			}
		}
	}

	return minfield;

}

}

}

}


