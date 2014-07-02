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
#include "nv_ip_gaussian.h"

#define KERNEL_OFFSET 2

void nv_gaussian5x5(nv_matrix_t *dest, int dch, const nv_matrix_t *src, int sch)
{
	static const float kernel[5][5] = {
		{ 2.969017E-03f, 1.330621E-02f, 2.193823E-02f, 1.330621E-02f, 2.969017E-03f },
		{ 1.330621E-02f, 5.963430E-02f, 9.832033E-02f, 5.963430E-02f, 1.330621E-02f },
		{ 2.193823E-02f, 9.832033E-02f, 1.621028E-01f, 9.832033E-02f, 2.193823E-02f },
		{ 1.330621E-02f, 5.963430E-02f, 9.832033E-02f, 5.963430E-02f, 1.330621E-02f },
		{ 2.969017E-03f, 1.330621E-02f, 2.193823E-02f, 1.330621E-02f, 2.969017E-03f }
	};
	int row;
	const int erow = dest->rows - KERNEL_OFFSET;
	const int ecol = dest->cols - KERNEL_OFFSET;
	int procs = nv_omp_procs();

	NV_ASSERT(dest->m == src->m);
#if 0
	if (scale == 0.0f) {
		float sum = 0.0f;
		int krow, kcol;
		for (krow = 0; krow < 5; ++krow) {
			for (kcol = 0; kcol < 5; ++kcol) {
				float gy = krow - 2.0f;
				float gx = kcol - 2.0f;
				float gaussian = expf(-(gx * gx) / 2.0f) * expf(-(gy * gy) / 2.0f);
				kernel[krow][kcol] = gaussian;
				sum += gaussian;
			}
		}
		scale = 1.0f / sum;
	}
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(procs)
#endif
	for (row = KERNEL_OFFSET; row < erow; ++row) {
		int col;
		for (col = KERNEL_OFFSET; col < ecol; ++col) {
			float v;
			v = 
				NV_MAT3D_V(src, row + 0 - KERNEL_OFFSET, col + 0 - KERNEL_OFFSET, sch) * kernel[0][0] +
				NV_MAT3D_V(src, row + 0 - KERNEL_OFFSET, col + 1 - KERNEL_OFFSET, sch) * kernel[0][1] +
				NV_MAT3D_V(src, row + 0 - KERNEL_OFFSET, col + 2 - KERNEL_OFFSET, sch) * kernel[0][2] +
				NV_MAT3D_V(src, row + 0 - KERNEL_OFFSET, col + 3 - KERNEL_OFFSET, sch) * kernel[0][3] +
				NV_MAT3D_V(src, row + 0 - KERNEL_OFFSET, col + 4 - KERNEL_OFFSET, sch) * kernel[0][4] +
				NV_MAT3D_V(src, row + 1 - KERNEL_OFFSET, col + 0 - KERNEL_OFFSET, sch) * kernel[1][0] +
				NV_MAT3D_V(src, row + 1 - KERNEL_OFFSET, col + 1 - KERNEL_OFFSET, sch) * kernel[1][1] +
				NV_MAT3D_V(src, row + 1 - KERNEL_OFFSET, col + 2 - KERNEL_OFFSET, sch) * kernel[1][2] +
				NV_MAT3D_V(src, row + 1 - KERNEL_OFFSET, col + 3 - KERNEL_OFFSET, sch) * kernel[1][3] +
				NV_MAT3D_V(src, row + 1 - KERNEL_OFFSET, col + 4 - KERNEL_OFFSET, sch) * kernel[1][4] +
				NV_MAT3D_V(src, row + 2 - KERNEL_OFFSET, col + 0 - KERNEL_OFFSET, sch) * kernel[2][0] +
				NV_MAT3D_V(src, row + 2 - KERNEL_OFFSET, col + 1 - KERNEL_OFFSET, sch) * kernel[2][1] +
				NV_MAT3D_V(src, row + 2 - KERNEL_OFFSET, col + 2 - KERNEL_OFFSET, sch) * kernel[2][2] +
				NV_MAT3D_V(src, row + 2 - KERNEL_OFFSET, col + 3 - KERNEL_OFFSET, sch) * kernel[2][3] +
				NV_MAT3D_V(src, row + 2 - KERNEL_OFFSET, col + 4 - KERNEL_OFFSET, sch) * kernel[2][4] +
				NV_MAT3D_V(src, row + 3 - KERNEL_OFFSET, col + 0 - KERNEL_OFFSET, sch) * kernel[3][0] +
				NV_MAT3D_V(src, row + 3 - KERNEL_OFFSET, col + 1 - KERNEL_OFFSET, sch) * kernel[3][1] +
				NV_MAT3D_V(src, row + 3 - KERNEL_OFFSET, col + 2 - KERNEL_OFFSET, sch) * kernel[3][2] +
				NV_MAT3D_V(src, row + 3 - KERNEL_OFFSET, col + 3 - KERNEL_OFFSET, sch) * kernel[3][3] +
				NV_MAT3D_V(src, row + 3 - KERNEL_OFFSET, col + 4 - KERNEL_OFFSET, sch) * kernel[3][4] +
				NV_MAT3D_V(src, row + 4 - KERNEL_OFFSET, col + 0 - KERNEL_OFFSET, sch) * kernel[4][0] +
				NV_MAT3D_V(src, row + 4 - KERNEL_OFFSET, col + 1 - KERNEL_OFFSET, sch) * kernel[4][1] +
				NV_MAT3D_V(src, row + 4 - KERNEL_OFFSET, col + 2 - KERNEL_OFFSET, sch) * kernel[4][2] +
				NV_MAT3D_V(src, row + 4 - KERNEL_OFFSET, col + 3 - KERNEL_OFFSET, sch) * kernel[4][3] +
				NV_MAT3D_V(src, row + 4 - KERNEL_OFFSET, col + 4 - KERNEL_OFFSET, sch) * kernel[4][4];
			
			NV_MAT3D_V(dest, row, col, dch) = v;//NV_MIN(v, 255.0f);
		}
	}
	for (row = 0; row < KERNEL_OFFSET && row < dest->rows; ++row) {
		int col;
		for (col = 0; col < dest->cols; ++col) {
			NV_MAT3D_V(dest, row, col, dch) = NV_MAT3D_V(src, row, col, sch);
		}
	}
	for (row = src->rows - KERNEL_OFFSET; row >= 0 && row < dest->rows; ++row) {
		int col;
		for (col = 0; col < dest->cols; ++col) {
			NV_MAT3D_V(dest, row, col, dch) = NV_MAT3D_V(src, row, col, sch);			
		}
	}
	for (row = 0; row < dest->rows; ++row) {
		int col;
		for (col = 0; col < KERNEL_OFFSET; ++col) {
			NV_MAT3D_V(dest, row, col, dch) = NV_MAT3D_V(src, row, col, sch);
		}
	}
	for (row = 0; row < dest->rows; ++row) {
		int col;
		for (col = src->cols - KERNEL_OFFSET; col < src->cols; ++col) {
			NV_MAT3D_V(dest, row, col, dch) = NV_MAT3D_V(src, row, col, sch);
		}
	}
}
