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
#include "distribution.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_gmap(DB * db, float tmin, float tmax, float bad_pix_val);
void calc_axis_gmap(DB * db, float tmin, float tmax, int axis);
void increment_gmap(DB * db, float * lmed, int axis);
void init_gmap(float * gmap, float * data, dim3 dsz, float bad_pix_val);

int d2h(float dval, float m_min, float m_max, int nbins);

}

}

}
#endif /* SRC_IMG_DSPK_GOODMAP_H_ */
