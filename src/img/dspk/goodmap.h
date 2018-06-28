/*
 * goodmap.h
 *
 *  Created on: Jun 28, 2018
 *      Author: byrdie
 */

#ifndef SRC_IMG_DSPK_GOODMAP_H_
#define SRC_IMG_DSPK_GOODMAP_H_

#include <cmath>

#include "util.h"
#include "median.h"
#include "threshold.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_gmap(DB * db, float tmin, float tmax, float bad_pix_val);
void calc_axis_gmap(DB * db, float tmin, float tmax, int axis);
void init_gmap(float * gmap, float * data, dim3 dsz, float bad_pix_val);

}

}

}
#endif /* SRC_IMG_DSPK_GOODMAP_H_ */
