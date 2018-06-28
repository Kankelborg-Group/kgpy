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
#include "goodmap.h"
#include "src/pyboost/pyboost.h"

namespace kgpy {

namespace img {

namespace dspk {

void dspk(DB * db, float tmin, float tmax, float bad_pix_val);
np::ndarray dspk_ndarr(np::ndarray & data, float thresh_min, float thresh_max, int kz, int ky, int kx, float bad_pix_val);



}

}

}

#endif /* SRC_IMG_DSPK_DSPK_H_ */
