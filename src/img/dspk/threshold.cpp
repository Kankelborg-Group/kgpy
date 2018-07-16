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

void calc_thresh(DB * db, float tmin, float tmax, int axis) {

	// start by calculating the exact threshold based off of histogram columns
	calc_exact_thresh(db, tmin, tmax, axis);

	// extrapolate threshold into statistically insignificant areas
	calc_extrap_thresh(db, tmin, tmax, axis);

}

void calc_exact_thresh(DB * db, float tmin, float tmax, int axis){

	float * cumd = db->cumd;
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

}

void calc_extrap_thresh(DB * db, float tmin, float tmax, int axis){

	// load from database
	float * t9 = db->t9;
	float * t1 = db->t1;

	// extrapolate threshold by slicing histogram along threshold
	int y1_min = median_extrapolation(db, t9, tmin, axis, -1);
	int y1_max = median_extrapolation(db, t1, tmax, axis, 1);

	// save the extrapolated trheshold curve to arrays
	apply_extrap_thresh(db, t9, tmin, y1_min, axis);
	apply_extrap_thresh(db, t1, tmax, y1_max, axis);




}

void apply_extrap_thresh(DB * db, float * t, float thresh, int y1, int axis){

	// load from database
	float * cnts = db->cnts;
	dim3 hsz = db->hsz;

	// compute threshold strides
	int tx = 1;
	int tz = tx * hsz.x;

	// first point at origin
	int x0 = 0;
	int y0 = 0;

	// find minimum counts for statistical significance
	int min_cnts = min_samples(thresh);

	int x1 = hsz.x - 1;

	// replace all statistically insignificant points with extrapolated line
#pragma acc parallel loop present(t), present(cnts)
	for(int x = 0; x < hsz.x; x++){

		float m = pts2slope(x0, y0, x1, y1);
		float b = pts2intercept(x0, y0, x1, y1);

		int T = tz * axis + tx * x;

		// if point is not statistically significant
		if(cnts[T] < min_cnts) {

			t[T] = m * x + b;	// make sure upper thresh does not cross lower thresh
			t[T] = fmax(fmin(t[T], hsz.y - 1), 0); // make sure we don't cross top/bottom of histogram

		}

	}

}

int median_extrapolation(DB * db, float * t, float thresh, int axis, int direction){

	// load from database
	float * cnts = db->cnts;
	dim3 hsz = db->hsz;
	dim3 tsz = db->tsz;

	// hold first point fixed
	int x0 = 0;
	int y0 = 0;

	// initial guess for second point
	int x1 = hsz.x - 1;
	int y1 = x1;

	// compute threshold strides
	int tx = 1;
	int tz = tx * tsz.x;

	// find minimum counts for statistical significance
	int min_cnts = min_samples(thresh);

	//#pragma acc update host(hist[0:hsz.xyz]), host(t[0:tsz.xyz])
	{
		// extrapolate slope of threshold
		while(true) {

			float lsum = 0;
			float usum = 0;

#pragma acc parallel loop reduction(+:usum,lsum) present(cnts), present(t)
			for(int x = 0; x < hsz.x; x++){

				int T = tz * axis + tx * x;
				float Y = t[T];

				float m = pts2slope(x0, y0, x1, y1);
				float b = pts2intercept(x0, y0, x1, y1);
				float y = m * x + b;

				if(cnts[T] > min_cnts) {
					if (Y > y) {
						usum++;
					} else {
						lsum++;
					}
				}

			}

			float ratio =  lsum / (usum + lsum);

			if (direction > 0) {
				if (ratio >= 0.50) {
					return y1;
				}
			} else {
				if (ratio <= 0.50) {
					return y1;
				}
			}

			y1 += direction;

		}
	}

}

int calc_hist_center(DB * db, int axis) {

	// load info from database
	float * cnts = db->cnts;
	dim3 tsz = db->tsz;
	int max_ind = 0;

	// compute threshold strides
	int tx = 1;
	int tz = tx * tsz.x;

	// find most statstically significant threshold
	float max_cnt = -10.0f;
#pragma acc update host(cnts[0:tsz.xyz])
	//#pragma acc data present(cnts[0:tsz.xyz])
	{
		//#pragma acc loop seq
		for(int X = 0; X < tsz.x; X++){

			int T = tz * axis + tx * X;

			//			printf("%f\n", cnts[T]);

			if(cnts[T] > max_cnt) {
				max_cnt = cnts[T];
				max_ind = X;
			}
		}

	}

	return max_ind;


}

float pts2slope(int x0, int y0, int x1, int y1) {

	float dy = (float) (y1 - y0);
	float dx = (float) (x1 - x0);

	return dy / dx;

}

float pts2intercept(int x0, int y0, int x1, int y1){

	float m = pts2slope(x0, y0, x1, y1);

	return y0 - m * x0;

}

int min_samples(float thresh){

	// compute how many counts are required for adequate statistics
	float interval = fmin(1.0f - thresh, thresh);
	float sigma = 10.0f;	// how many points need to be outside the confidence interval
	int min_cnts = sigma / interval;

	return min_cnts;

}

}

}

}

