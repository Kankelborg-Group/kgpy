/*
 * distribution.h
 *
 *  Created on: Jun 28, 2018
 *      Author: byrdie
 */

#ifndef SRC_IMG_DSPK_DISTRIBUTION_H_
#define SRC_IMG_DSPK_DISTRIBUTION_H_

#include "util.h"

namespace kgpy {

namespace img {

namespace dspk {

void calc_histogram(DB * db, float * lmed, int axis);
void calc_cumulative_distribution(DB * db, int axis);

}

}

}


#endif /* SRC_IMG_DSPK_DISTRIBUTION_H_ */
