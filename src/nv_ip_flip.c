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

#include "nv_core.h"
#include "nv_ip.h"

void 
nv_flip_x(nv_matrix_t *flip, const nv_matrix_t *src)
{
	int x, y, i;

	NV_ASSERT(flip->rows == src->rows);
	NV_ASSERT(flip->cols == src->cols);
	NV_ASSERT(flip->n == src->n);

	for (y = 0; y < src->rows; ++y) {
		for (x = 0; x < src->cols; ++x) {
			for (i = 0; i < src->n; ++i) {
				NV_MAT3D_V(flip, y, src->cols - x - 1, i) = NV_MAT3D_V(src, y, x, i);
			}
		}
	}
}

void 
nv_flip_y(nv_matrix_t *flip, const nv_matrix_t *src)
{
	int x, y, i;

	NV_ASSERT(flip->rows == src->rows);
	NV_ASSERT(flip->cols == src->cols);
	NV_ASSERT(flip->n == src->n);

	for (y = 0; y < src->rows; ++y) {
		for (x = 0; x < src->cols; ++x) {
			for (i = 0; i < src->n; ++i) {
				NV_MAT3D_V(flip, src->rows - y - 1, x, i) = NV_MAT3D_V(src, y, x, i);
			}
		}
	}
}

