/*
 * util.h
 *
 *  Created on: Jun 26, 2018
 *      Author: byrdie
 */

#ifndef SRC_IMG_DSPK_UTIL_H_
#define SRC_IMG_DSPK_UTIL_H_

namespace kgpy {

namespace img {

namespace dspk {

class dim3 {
public:
	int x;
	int y;
	int z;
	int xyz;

	dim3(int x, int y, int z);
	dim3 operator+(dim3 right);
	dim3 operator*(dim3 right);
	dim3 operator/(int right);
};

class DB {
public:
	dim3 dsz;	// shape of data array
	dim3 ksz;	// shape of local median kernel;
	float * data;	// input data array
	float * lmed;	// local median of data array
	float * gmap;	// map of good pixels

	DB(float * data, dim3 dsz, dim3 ksz);
};

}

}

}


#endif /* SRC_IMG_DSPK_UTIL_H_ */
