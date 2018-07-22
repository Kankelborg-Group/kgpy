/*
 * badpix.cpp
 *
 *  Created on: Jul 19, 2018
 *      Author: byrdie
 */

#include "badpix.h"

namespace kgpy {

namespace img {

namespace dspk {

void fix_badpix(DB * db){

	dim3 dsz = db->dsz;

	float * conv = new float[dsz.xyz];
#pragma acc data create(conv[0:dsz.xyz])
	{
		convol_goodpix(db, conv);
		replace_badpix(db, conv);
	}

}

void convol_goodpix(DB * db, float * conv){

	float * data = db->data;
	dim3 dsz = db->dsz;

	float * tmp = new float[dsz.xyz];
#pragma acc data create(tmp[0:dsz.xyz])
	{
		convol_goodpix_axis(db, data, conv, 0);
		convol_goodpix_axis(db, conv, tmp, 1);
		convol_goodpix_axis(db, tmp, conv, 2);

	}

}

void convol_goodpix_axis(DB * db, float * input, float * output, int axis){

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
	int kx = ksz.x;
	int ky = ksz.y;
	int kz = ksz.z;

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

#pragma acc parallel loop collapse(3) present(gmap), present(input), present(output)
	for(int z = 0; z < dz; z++){	// loop along z axis of data array)
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
					float u = input[I];

					// kernel value at this kernel location
					float v = kernel(i);

					// execute convolution
					sum += g * u * v;
					norm += g * v;

				}

				if(norm > 0.0f) {
					output[L] = sum / norm;	// store results of convolution
				} else {
					output[L] = 0.0f;
				}


			}

		}

	}

}

void replace_badpix(DB * db, float * conv){

	float * data = db->data;
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

#pragma acc parallel loop collapse(3) present(gmap), present(data), present(conv)
	for(int z = 0; z < dz; z++){	// loop along z axis of data array)
		for(int y = 0; y < dy; y++){	// loop along y axis of of data array
			for(int x = 0; x < dx; x++){	// loop along x axis of of data array

				// overall linear index of data array
				int L =  (sz * z) + (sy * y) + (sx * x);

				if(gmap[L] == 0.0f) {
					data[L] = conv[L];
				}

			}
		}
	}

}

float kernel(int x){

	// cast array index to floating point
	float X = -(float) abs(x);

	// length scale of kernel
	float a = 0.05;

	return exp(X / a);


}


}

}

}

