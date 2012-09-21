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

#ifndef NV_IP_EUCLIDEAN_COLOR_H
#define NV_IP_EUCLIDEAN_COLOR_H

#ifdef __cplusplus
extern "C" {
#endif
#include "nv_core.h"

void nv_color_bgr2euclidean(nv_matrix_t *ec, const nv_matrix_t *bgr);
void nv_color_bgr2euclidean_scalar(nv_matrix_t *ec, int ec_m, const nv_matrix_t *bgr, int bgr_m);

void nv_color_euclidean2bgr_scalar(nv_matrix_t *bgr, int bgr_m, const nv_matrix_t *ec, int ec_m);
void nv_color_euclidean2bgr(nv_matrix_t *bgr, const nv_matrix_t *ec);

#ifdef __cplusplus
}
#endif

#endif
