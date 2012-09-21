/*
 * This file is part of libnv.
 *
 * Copyright (C) 2008-2012 nagadomi@nurs.or.jp
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License,
 * or any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef NV_IP_CCV_H
#define NV_IP_CCV_H

#ifdef __cplusplus
extern "C" {
#endif
#include "nv_core.h"

#define NV_CCV_THRESHOLD(rows, cols) NV_ROUND_INT(sqrtf(rows * cols) * 0.1f)
#define NV_CCV_DIM(k) (k * 2)

void nv_ccv_nn(nv_matrix_t *ccv,
			   int ccv_j,
			   const nv_matrix_t *image,
			   const nv_matrix_t *centroids,
			   int threshold
	);

#ifdef __cplusplus
}
#endif

#endif
