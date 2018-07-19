/*
 * badpix.h
 *
 *  Created on: Jul 19, 2018
 *      Author: byrdie
 */

#ifndef SRC_IMG_DSPK_BADPIX_H_
#define SRC_IMG_DSPK_BADPIX_H_

#include <math.h>

#include "util.h"

namespace kgpy {

namespace img {

namespace dspk {

void fix_badpix(DB * db);
void convol_goodpix(DB * db, float * conv);
void convol_goodpix_axis(DB * db, float * input, float * output, int axis);
void replace_badpix(DB * db, float * conv);

float kernel(int x);


}

}

}

#endif /* SRC_IMG_DSPK_BADPIX_H_ */
