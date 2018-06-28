/*
 * threshold.h
 *
 *  Created on: Jun 28, 2018
 *      Author: byrdie
 */

#ifndef SRC_IMG_DSPK_THRESHOLD_H_
#define SRC_IMG_DSPK_THRESHOLD_H_

#include "util.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_thresh(float * thresh, float * hist, float * data, dim3 tsz, dim3 hsz, dim3 dsz, float percentile, int axis);

}

}

}

#endif /* SRC_IMG_DSPK_THRESHOLD_H_ */
