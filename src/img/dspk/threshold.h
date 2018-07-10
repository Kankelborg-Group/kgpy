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

}

}

}

#endif /* SRC_IMG_DSPK_THRESHOLD_H_ */
