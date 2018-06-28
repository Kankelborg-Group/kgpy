/*
 * dspk.h
 *
 *  Created on: Jun 26, 2018
 *      Author: byrdie
 */

#ifndef SRC_IMG_DSPK_DSPK_H_
#define SRC_IMG_DSPK_DSPK_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <string>

#include "util.h"
#include "median.h"
#include "src/pyboost/pyboost.h"

namespace kgpy {

namespace img {

namespace dspk {

const dim3 xhat(1, 0, 0);
const dim3 yhat(0, 1, 0);
const dim3 zhat(0, 0, 1);

void dspk(DB * db, float tmin, float tmax);
np::ndarray dspk_ndarr(np::ndarray & data, float thresh_min, float thresh_max, int kz, int ky, int kx);



}

}

}

#endif /* SRC_IMG_DSPK_DSPK_H_ */
