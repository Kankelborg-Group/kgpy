/*
 * threshold.h
 *
 *  Created on: Jun 28, 2018
 *      Author: byrdie
 */

#ifndef KGPY_IMG_DSPK_THRESHOLD_H_
#define KGPY_IMG_DSPK_THRESHOLD_H_

#include <kgpy/img/dspk/util.h>
#include<math.h>


namespace kgpy {

namespace img {

namespace dspk {

void calc_thresh(DB * db, float tmin, float tmax, int axis);
void calc_exact_thresh(DB * db, float tmin, float tmax, int axis);
void calc_extrap_thresh(DB * db, float tmin, float tmax, int axis);
void apply_extrap_thresh(DB * db, float * t, float thresh, int x0, float theta, int axis);
float median_extrapolation(DB * db, float * t, float thresh, int x0, int axis);
void calc_intensity_thresh(DB * db, float tmin, float tmax);

int calc_hist_center(DB * db, int axis);
float pts2slope(int x0, int y0, int x1, int y1);
float calc_intercept(int x0, int y0, float m);
int min_samples(float thresh);

}

}

}

#endif /* KGPY_IMG_DSPK_THRESHOLD_H_ */
