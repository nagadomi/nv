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

#include "nv_ip_patch.h"

nv_matrix_t *
nv_patch_matrix_alloc(nv_matrix_t *src, int patch_size)
{
	NV_ASSERT(patch_size <= src->rows);
	NV_ASSERT(patch_size <= src->cols);
	
	return nv_matrix3d_alloc(src->n * patch_size * patch_size,
							 src->rows - patch_size,
							 src->cols - patch_size);
}

void
nv_patch_extract(nv_matrix_t *patches,
				 const nv_matrix_t *src, int patch_size)
{
	int y;
	
	NV_ASSERT(patches->n == src->n * patch_size * patch_size);
	NV_ASSERT(patches->rows == src->rows - patch_size);
	NV_ASSERT(patches->cols == src->cols - patch_size);
	
	for (y = 0; y < patches->rows; ++y) {
		int x;
		for (x = 0; x < patches->cols; ++x) {
			int h;
			for (h = 0; h < patch_size; ++h) {
				int w;
				for (w = 0; w < patch_size; ++w) {
					int c = 0;
					for (c = 0; c < src->n; ++c) {
						NV_MAT3D_V(patches, y, x, (h * patch_size + w) * src->n + c) = NV_MAT3D_V(src, y + h, x + w, c);
					}
				}
			}
		}
	}
}

