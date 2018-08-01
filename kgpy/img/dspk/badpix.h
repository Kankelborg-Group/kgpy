/*
 * badpix.h
 *
 *  Created on: Jul 19, 2018
 *      Author: byrdie
 */

#ifndef KGPY_IMG_DSPK_BADPIX_H_
#define KGPY_IMG_DSPK_BADPIX_H_

#include <kgpy/img/dspk/util.h>
#include <math.h>


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

#endif /* KGPY_IMG_DSPK_BADPIX_H_ */
