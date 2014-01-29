/*
 * This file is part of libnv.
 *
 * Copyright (C) 2014 nagadomi@nurs.or.jp
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

#ifndef NV_FV_RECTANGLE_FEATURE_H
#define NV_FV_RECTANGLE_FEATURE_H

#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NV_RECTANGLE_FEATURE_N 1152

void nv_rectangle_feature(nv_matrix_t *fv, 
						  int fv_j,
						  const nv_matrix_t *integral_image,
						  int x, int y, int width, int height);

#ifdef __cplusplus
}
#endif

#endif
