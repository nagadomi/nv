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

#ifndef NV_IP_INTEGRAL_H
#define NV_IP_INTEGRAL_H
#include "nv_core.h"
#ifdef __cplusplus
extern "C" {
#endif

#define NV_INTEGRAL_V(sum, x, y, xw, yh) \
	(NV_MAT3D_V((sum), (y), (x), 0) \
	- NV_MAT3D_V((sum), (y), (xw), 0) \
	- NV_MAT3D_V((sum), (yh), (x), 0) \
	+ NV_MAT3D_V((sum), (yh), (xw), 0)) 

void nv_integral(nv_matrix_t *integral, const nv_matrix_t *img, int channel);
void nv_integral_tilted(nv_matrix_t *integral,
						const nv_matrix_t *img, int channel);

#ifdef __cplusplus
}
#endif


#endif
