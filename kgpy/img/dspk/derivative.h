/*
 * derivative.h
 *
 *  Created on: Jul 20, 2018
 *      Author: byrdie
 */

#ifndef SRC_IMG_DSPK_DERIVATIVE_H_
#define SRC_IMG_DSPK_DERIVATIVE_H_

#include <math.h>

#include "util.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_2nd_deriv(DB * db, float * deriv, int axis);

float kernel_2nd_deriv(int i);

}

}

}


#endif /* SRC_IMG_DSPK_DERIVATIVE_H_ */
