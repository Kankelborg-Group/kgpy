/*
 * goodmap.h
 *
 *  Created on: Jun 28, 2018
 *      Author: byrdie
 */

#ifndef KGPY_IMG_DSPK_GOODMAP_H_
#define KGPY_IMG_DSPK_GOODMAP_H_

#include <kgpy/img/dspk/distribution.h>
#include <kgpy/img/dspk/median.h>
#include <kgpy/img/dspk/threshold.h>
#include <kgpy/img/dspk/util.h>
#include <cmath>
#include <math.h>


namespace kgpy {

namespace img {

namespace dspk {

void calc_gmap(DB * db, float tmin, float tmax, float bad_pix_val);
void calc_axis_gmap(DB * db, float tmin, float tmax, int axis);
void finalize_gmap(DB * db);
void increment_gmap(DB * db, float * lmed, int axis);
void init_gmap(float * gmap, float * data, dim3 dsz, float bad_pix_val);

int d2h(float dval, float m_min, float m_max, int nbins);

}

}

}
#endif /* KGPY_IMG_DSPK_GOODMAP_H_ */
