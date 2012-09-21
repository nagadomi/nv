/*
 * This file is part of libnv.
 *
 * Copyright (C) 2012 nagadomi@nurs.or.jp
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

#ifndef NV_IP_MORPHOLOGY_H
#define NV_IP_MORPHOLOGY_H

#ifdef __cplusplus
extern "C" {
#endif

/* 3x3 rectangle */
void nv_erode(nv_matrix_t *dst, int dch, const nv_matrix_t *src, int sch);
void nv_dilate(nv_matrix_t *dst, int dch, const nv_matrix_t *src, int sch);
void nv_morph_close(nv_matrix_t *dst, int dch, const nv_matrix_t *src, int sch);
void nv_morph_open(nv_matrix_t *dst, int dch, const nv_matrix_t *src, int sch);

/* not implemented
typedef enum {
	NV_MORPHOLOGY_FILTER_RECTANGLE = 0
} nv_morphology_filter_type_e;

void nv_erode_ex(nv_matrix_t *dst, const nv_matrix_t *src,
				 nv_morphology_filter_type_e type,
				 int size);
void nv_dilate_ex(nv_matrix_t *dst, const nv_matrix_t *src,
				  nv_morphology_filter_type_e type,
				  int size);
*/

#ifdef __cplusplus
}
#endif

#endif

