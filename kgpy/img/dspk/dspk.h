/*
 * dspk.h
 *
 *  Created on: Jun 26, 2018
 *      Author: byrdie
 */

#ifndef KGPY_IMG_DSPK_DSPK_H_
#define KGPY_IMG_DSPK_DSPK_H_

#include <kgpy/img/dspk/badpix.h>
#include <kgpy/img/dspk/goodmap.h>
#include <kgpy/img/dspk/util.h>
#include <kgpy/pyboost/pyboost.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <string>


namespace kgpy {

namespace img {

namespace dspk {

void dspk(DB * db, float tmin, float tmax, float bad_pix_val);
py::tuple dspk_ndarr(np::ndarray & data, float thresh_min, float thresh_max, int kz, int ky, int kx, float bad_pix_val);
void dspk_idl(float * data, float thresh_min, float thresh_max, int dz, int dy, int dx,int kz, int ky, int kx, float bad_pix_val);



}

}

}

#endif /* KGPY_IMG_DSPK_DSPK_H_ */
