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

	// compute histogram strides
	int hx = 1;
	int hy = hx * hsz.x;
	int hz = hy * hsz.y;

	// compute threshold strides
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
#pragma acc loop seq
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
#pragma acc loop seq
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





	// replace all statistically insignificant points with extrapolated line
#pragma acc parallel loop present(t1), present(t9), present(cnts)
	for(int x = 0; x < hsz.x; x++){

		// find most statstically significant threshold
#pragma acc loop seq
		int max_cnt = 0;
		for(int X = 0; X < hsz.x; X++){

			int T = tz * axis + tx * X;

			//		printf("%d\n", cnts[T]);

			if(cnts[T] > max_cnt) {
				max_cnt = cnts[T];
				cnt9 = X;
			}

		}

		// find highest statisically significant threshold
#pragma acc loop seq
		for(int X = hsz.x - 1; X >= 0; X--){

			int T = tz * axis + tx * X;

			if(cnts[T] > min_cnts) {
				cnt1 = X;
				break;
			}

		}

		// calculate line between these two points
		int a = tz * axis + tx * cnt1;
		int b = tz * axis + tx * cnt9;
		float m1 = (t1[a] - t1[b]) / ((float)(cnt1 - cnt9));
		float m9 = (t9[a] - t9[b]) / ((float)(cnt1 - cnt9));
		float y1 = t1[a] - (m1 * cnt1);
		float y9 = t9[a] - (m9 * cnt1);


		int T = tz * axis + tx * x;

		// if point is not statistically significant
		if(cnts[T] < min_cnts) {

			t1[T] = fmax(m1 * x + y1, x + 1);	// make sure upper thresh does not cross lower thresh
			t1[T] = fmax(fmin(t1[T], hsz.y - 1), 0);

			t9[T] = fmin(m9 * x + y9, x - 1);	// make sure lower thresh does not cross upper thresh
			t9[T] = fmax(fmin(t9[T], hsz.y - 1), 0);

		}

	}

}

}

}

}

