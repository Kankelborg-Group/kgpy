/*
 * median.cpp
 *
 *  Created on: Jun 26, 2018
 *      Author: byrdie
 */

#include "median.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_local_median(float * lmed, float * data, float * gmap, dim3 dsz, dim3 ksz, dim3 axis){

	// split data size into single variables
	int dx = dsz.x;
	int dy = dsz.y;
	int dz = dsz.z;

	// split axis into single variables
	int ax = axis.x;
	int ay = axis.y;
	int az = axis.z;

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

	printf("%d %d %d\n", sx, sy, sz);

	// data size along current axis
	int dX = (dx * ax) + (dy * ay) + (dz * az);

	printf("%d\n", dX);

	// halo size along current axis
	int hX = (hx * ax) + (hy * ay) + (hz * az);

#pragma acc data copyin(data[0:dsz.xyz]), copyout(lmed[0:dsz.xyz])
#pragma acc kernels
#pragma acc loop independent gang(dz)
	for(int z = 0; z < dz; z++){	// loop along z axis of data/median array
#pragma acc loop independent gang(dy)
		for(int y = 0; y < dy; y++){	// loop along y axis of of data/median array
#pragma acc loop independent gang(dx/32 + 1), vector(32)
			for(int x = 0; x < dx; x++){	// loop along x axis of of data/median array

				// variable to store if median was located
				bool median_found = false;

				// overall linear index of data/median array
				int L =  (sz * z) + (sy * y) + (sx * x);

//				printf("%d\n", L);

				// 3D index of data/median array along current axis
				int X = (x * ax) + (y * ay) + (z * az);

				// loop along current axis to select a trial median value
				for(int i = -hX; i <= hX; i++){

					// check if we're inside bounds of the data/median array
					if ((X + i) > (dX - 1) or (X + i) < 0) continue;

					// Compute linear index of trial median value
					int I = sz * (z + az * i) + sy * (y + ay * i) + sx * (x + ax * i);

					// store trial median value
					float u = data[I];

					// initialize bins to store how many values are smaller/equal/larger than trial value
					int sm = 0;
					int eq = 0;
					int lg = 0;

					// loop along current axis to compare trial value to all other values within kernel
					for (int j = -hX; j <= hX; j++){

						// check if we're inside bounds of the data/median array
						if ((X + j) > (dX - 1) or (X + j) < 0) continue;

						// compute linear index of comparison median value
						int J = sz * (z + az * j) + sy * (y + ay * j) + sx * (x + ax * j);

						// check if the comparison value is valid data
//						if (gmap[J] == 0.0f) continue;

						// store comparison median value
						float v = data[J];

						// increment counts for each bin
						if (u > v) {
							sm = sm + 1;
						} else if (u == v) {
							eq = eq + 1;
						} else {
							lg = lg + 1;
						}



					}

					// calculate total count
					int tot = sm + eq + lg;

					// calculate index of median
					int M = tot / 2;

					// check if median value and store in output if so
					if ((sm - 1) < M) {
						if ((sm + eq - 1) >= M){

							lmed[L] = u;
							median_found = true;

						}
					}


				}


				if (not median_found) {
					lmed[L] = 0.0f;
				}


			}

		}

	}

}

}

}

}
