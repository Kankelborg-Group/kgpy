/*
 * derivative.cpp
 *
 *  Created on: Jul 20, 2018
 *      Author: byrdie
 */

#include "derivative.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_2nd_deriv(DB * db, float * deriv, int axis){

	float * data = db->data;
	float * gmap = db->gmap;
	dim3 dsz = db->dsz;
	dim3 ksz = db->ksz;

	// split data size into single variables
	int dx = dsz.x;
	int dy = dsz.y;
	int dz = dsz.z;

	// split axis into single variables
	dim3 uvec = axes[axis];
	int ax = uvec.x;
	int ay = uvec.y;
	int az = uvec.z;

	// split kernel into single variables
	int kx = 3;
	int ky = 3;
	int kz = 3;

	// calculate halo size from kernel size
	int hx = kx / 2;
	int hy = ky / 2;
	int hz = kz / 2;

	// compute array strides in each dimension
	int sx = 1;
	int sy = sx * dx;
	int sz = sy * dy;

	// data size along current axis
	int dX = (dx * ax) + (dy * ay) + (dz * az);

	// halo size along current axis
	int hX = (hx * ax) + (hy * ay) + (hz * az);

#pragma acc parallel loop collapse(3) present(gmap), present(data), present(deriv)
	for(int z = 0; z < dz; z++){	// loop along z axis of data array
		for(int y = 0; y < dy; y++){	// loop along y axis of of data array
			for(int x = 0; x < dx; x++){	// loop along x axis of of data array

				// overall linear index of data array
				int L =  (sz * z) + (sy * y) + (sx * x);

				// 3D index of data/median array along current axis
				int X = (x * ax) + (y * ay) + (z * az);

				float sum = 0.0f;
				float norm = 0.0f;

#pragma acc loop seq
				for(int i = -hX; i <= hX; i++){ // convolution loop

					// check if we're inside bounds of the data array
					if ((X + i) > (dX - 1) or (X + i) < 0) continue;

					// Compute linear index of trial median value
					int I = sz * (z + az * i) + sy * (y + ay * i) + sx * (x + ax * i);

					// goodmap value at this kernel location
					float g = gmap[I];

					// data value at this kernel location
					float u = data[I];

					// kernel value at this kernel location
					float v = kernel_2nd_deriv(i);

					// execute convolution
					sum += g * u * v;
					norm += 1;

				}

				deriv[L] = sum;	// store results of convolution

//				if(norm > 0.0f) {
//					deriv[L] = sum / norm;	// store results of convolution
//				} else {
//					deriv[L] = 0.0f;
//				}


			}
		}
	}

}

float kernel_2nd_deriv(int i){

	float ret;

	if(i == -1) {
		ret = 0.5f;
	} else if(i == 0) {
		ret = -2.0f;
	} else if(i == 1) {
		ret = 0.5f;
	}

	return ret;

}

}

}

}


