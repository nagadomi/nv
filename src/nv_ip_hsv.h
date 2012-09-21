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

#ifndef NV_IP_HSV_H
#define NV_IP_HSV_H

#ifdef __cplusplus
extern "C" {
#endif
#include "nv_core.h"

void nv_color_bgr2hsv(nv_matrix_t *hsv, const nv_matrix_t *bgr);
void nv_color_bgr2hsv_scalar(nv_matrix_t *hsv, int hsv_j,
							 const nv_matrix_t *bgr, int bgr_j);

void nv_color_hsv2bgr_scalar(nv_matrix_t *bgr, int bgr_j,
							 const nv_matrix_t *hsv, int hsv_j);
void nv_color_hsv2bgr(nv_matrix_t *bgr, const nv_matrix_t *hsv);

#define NV_CH_H 0
#define NV_CH_S 1
#define NV_CH_V	2

#ifdef __cplusplus
}
#endif

#endif
