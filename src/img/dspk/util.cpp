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

dim3::dim3(int x, int y, int z) : x(x), y(y), z(z) {

	xyz = x * y * z;

}

dim3::operator+(dim3)

DB::DB(float * data, dim3 dsz, dim3 ksz) : data(data), dsz(dsz), ksz(ksz) {

	lmed = new float[dsz.xyz];	// allocate local median array
	gmap = new float[dsz.xyz];	// allocate good pixel map array

}

}

}

}


