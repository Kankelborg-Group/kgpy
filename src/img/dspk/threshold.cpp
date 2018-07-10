/*
 * threshold.cpp
 *
 *  Created on: Jun 28, 2018
 *      Author: byrdie
 */

#include "threshold.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_thresh(DB * db, float tmin, float tmax, int axis){

	float * cumd = db->cumd;
	float * cnts = db->cnts;
	float * t1 = db->t1;
	float * t9 = db->t9;
	dim3 hsz = db->hsz;
	dim3 tsz = db->tsz;
	float dmax = db->dmax;
	float dmin = db->dmin;


	int hx = 1;
	int hy = hx * hsz.x;
	int hz = hy * hsz.y;

	int tx = 1;
	int tz = tx * hsz.x;

	// determine initial threshold array, ignoring counts
#pragma acc parallel loop present(cumd), present(t1), present(t9)
	for(int x = 0; x < hsz.x; x++){

		// index of threshold array
		int T = tz * axis + tx * x;

		//initialize thresholds
		t1[T] = x;
		t9[T] = x - 1;

		// locate lower threshold
		int y;
		for(y = 0; y < hsz.y; y++){

			// check if above lower threshold
			int H = hz * axis + hy * y + hx * x;
			float c = cumd[H];
			if (c > tmin) {
				t9[T] = y - 1;
				break;
			}

		}

		// locate upper threshold, starting from where we left off in the last loop
		for(int Y = y; Y < hsz.y; Y++){

			// check if above lower threshold
			int H = hz * axis + hy * Y + hx * x;
			float c = cumd[H];
			if (c > tmax) {
				t1[T] = Y;
				break;
			}

		}

	}

	// extrapolate threshold arrays in areas with inadequate statistics
	float thresh = fmin(tmin, 1.0f - tmax);
	float sigma = 10.0f;	// how many points need to be outside the confidence interval
	int min_cnts = sigma / thresh;
	int cnt9 = 1;
	int cnt1 = 0;

	printf("____________________________________\n");

	// find most statstically significant threshold
	//#pragma acc update host(cnts[0:tsz.xyz]), host(t1[0:tsz.xyz]), host(t9[0:tsz.xyz])
#pragma acc data present(cnts)
	{
#pragma acc loop seq
		int max_cnt = 0;
		for(int x = 0; x < hsz.x; x++){

			int T = tz * axis + tx * x;

			//		printf("%d\n", cnts[T]);

			if(cnts[T] > max_cnt) {
				max_cnt = cnts[T];
				cnt9 = x;
			}

		}

		// find highest statisically significant threshold
#pragma acc loop seq
		for(int x = hsz.x - 1; x >= 0; x--){

			int T = tz * axis + tx * x;

			if(cnts[T] > min_cnts) {
				cnt1 = x;
				break;
			}

		}
	}

	// calculate line between these two points
	int a = tz * axis + tx * cnt1;
	int b = tz * axis + tx * cnt9;
	float m1 = (t1[a] - t1[b]) / ((float)(cnt1 - cnt9));
	float m9 = (t9[a] - t9[b]) / ((float)(cnt1 - cnt9));
	float y1 = t1[a] - (m1 * cnt1);
	float y9 = t9[a] - (m9 * cnt1);

	printf("%d %d\n", cnt1, cnt9);
	printf("%f %f\n", m1, y1);
	printf("%f %f\n", m9, y9);
	printf("%d\n", axis);

	// replace all statistically insignificant points with extrapolated line
#pragma acc parallel loop present(t1), present(t9), present(cnts)
	for(int x = 0; x < hsz.x; x++){

		int T = tz * axis + tx * x;

		// if point is not statistically significant
		if(cnts[T] < min_cnts) {

			t1[T] = fmax(fmin(m1 * x + y1, hsz.y - 1), 0);
			t9[T] = fmax(fmin(m9 * x + y9, hsz.y - 1), 0);

		}

	}
	//#pragma acc update device(t1[0:tsz.xyz]), device(t9[0:tsz.xyz])

}

}

}

}

