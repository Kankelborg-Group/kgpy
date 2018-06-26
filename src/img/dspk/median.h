/*
 * median.h
 *
 *  Created on: Jun 26, 2018
 *      Author: byrdie
 */

#ifndef SRC_IMG_DSPK_MEDIAN_H_
#define SRC_IMG_DSPK_MEDIAN_H_

#include "util.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_local_median(float * lmed, float * data, float * gmap, dim3 dsz, dim3 ksz, dim3 axis);

}

}

}


#endif /* SRC_IMG_DSPK_MEDIAN_H_ */
