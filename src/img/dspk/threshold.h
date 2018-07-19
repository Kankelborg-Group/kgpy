/*
 * threshold.h
 *
 *  Created on: Jun 28, 2018
 *      Author: byrdie
 */

#ifndef SRC_IMG_DSPK_THRESHOLD_H_
#define SRC_IMG_DSPK_THRESHOLD_H_

#include<math.h>

#include "util.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_thresh(DB * db, float tmin, float tmax, int axis);
void calc_exact_thresh(DB * db, float tmin, float tmax, int axis);
void calc_extrap_thresh(DB * db, float tmin, float tmax, int axis);
void apply_extrap_thresh(DB * db, float * t, float thresh, int x0, int y1, int axis);
int median_extrapolation(DB * db, float * t, float thresh, int x0, int axis, int direction);
void calc_intensity_thresh(DB * db, float tmin, float tmax);

int calc_hist_center(DB * db, int axis);
float pts2slope(int x0, int y0, int x1, int y1);
float pts2intercept(int x0, int y0, int x1, int y1);
int min_samples(float thresh);

}

}

}

#endif /* SRC_IMG_DSPK_THRESHOLD_H_ */
