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
#include "nv_ip_integral.h"

/*
 * Integral Image
 * 積分画像
 */
void 
nv_integral(nv_matrix_t *integral,
			const nv_matrix_t *img, int channel)
{
	int row, col;
	int erow = img->rows + 1;
	int ecol = img->cols + 1;

	NV_ASSERT(
		integral->rows - 1 == img->rows 
		&& integral->cols - 1 == img->cols
	);

	nv_matrix_zero(integral);
	for (row = 1; row < erow; ++row) {
		float col_sum = 0.0f;
		for (col = 1; col < ecol; ++col) {
			float col_val = NV_MAT3D_V(img, row - 1, col - 1, channel);
			NV_MAT3D_V(integral, row, col, 0) =	
				NV_MAT3D_V(integral, row - 1, col, 0) + col_sum + col_val;
			col_sum += col_val;
		}
	}
}

/*
 * 45°回転したIntegral Image
 */
void 
nv_integral_tilted(nv_matrix_t *integral,
				   const nv_matrix_t *img, int channel)
{
	int row, col, scol, srow;
	int erow = img->rows + 1;
	int ecol = img->cols + 1;
	nv_matrix_t *prev_tilted = nv_matrix_alloc(img->cols + 1, 1);

	NV_ASSERT(
		integral->rows - 1 == img->rows 
		&& integral->cols - 1 == img->cols
	);

	nv_matrix_zero(prev_tilted);
	nv_matrix_zero(integral);

	for (scol = img->cols; scol > 0; --scol) {
		float tilted_sum = 0.0f;
		for (row = 1, col = scol; row < erow && col < ecol; ++row, ++col) {
			float tilted_val = NV_MAT3D_V(img, row - 1, col - 1, channel);
			if (col + 1 == ecol) {
				NV_MAT3D_V(integral, row, col, 0) = 
					NV_MAT3D_V(integral, row - 1, col, 0)
					+ tilted_sum + tilted_val;
			} else {
				NV_MAT3D_V(integral, row, col, 0) = 
					NV_MAT3D_V(integral, row - 1, col + 1, 0) 
					+ NV_MAT_V(prev_tilted, 0, col)
					+ tilted_sum + tilted_val;
			}
			tilted_sum += tilted_val;
			NV_MAT_V(prev_tilted, 0, col) = tilted_sum;
		}
	}
	for (srow = 2; srow < erow; ++srow) {
		float tilted_sum = 0.0f;
		for (row = srow, col = 1; row < erow && col < ecol; ++row, ++col) {
			float tilted_val = NV_MAT3D_V(img, row - 1, col - 1, channel);
			if (col + 1 == ecol) {
				NV_MAT3D_V(integral, row, col, 0) = 
					NV_MAT3D_V(integral, row - 1, col, 0)
					+ tilted_sum + tilted_val;
			} else {
				NV_MAT3D_V(integral, row, col, 0) = 
					NV_MAT3D_V(integral, row - 1, col + 1, 0) 
					+ NV_MAT_V(prev_tilted, 0, col)
					+ tilted_sum + tilted_val;
			}
			tilted_sum += tilted_val;
			NV_MAT_V(prev_tilted, 0, col) = tilted_sum;
		}
	}

	nv_matrix_free(&prev_tilted);
}

