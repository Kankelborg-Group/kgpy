/*
 * median.h
 *
 *  Created on: Jun 26, 2018
 *      Author: byrdie
 */

#ifndef KGPY_IMG_DSPK_MEDIAN_H_
#define KGPY_IMG_DSPK_MEDIAN_H_


#include <kgpy/img/dspk/util.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <string>


namespace kgpy {

namespace img {

namespace dspk {

void calc_local_median(float * lmed, float * data, float * gmap, dim3 dsz, dim3 ksz, int axis);

}

}

}


#endif /* KGPY_IMG_DSPK_MEDIAN_H_ */
