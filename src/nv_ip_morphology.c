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

#include "nv_core.h"
#include "nv_ip_morphology.h"

/* 3x3 モルフォロジー演算 */

#define KERNEL3X3_OFFSET 1

void
nv_erode(nv_matrix_t *dest, int dch, const nv_matrix_t *src, int sch)
{
	int row;
	
	NV_ASSERT(dest->rows == src->rows && dest->cols == dest->cols);
	
#ifdef _OPENMP
#pragma omp parallel for
#endif	
	for (row = KERNEL3X3_OFFSET; row < dest->rows - KERNEL3X3_OFFSET; ++row) {
		int col;
		for (col = KERNEL3X3_OFFSET; col < dest->cols - KERNEL3X3_OFFSET; ++col) {
			int i;
			float min_v = FLT_MAX;
			float v[9];
			
			v[0] = NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch);
			v[1] = NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch);
			v[2] = NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch);
			v[3] = NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch);
			v[4] = NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch);
			v[5] = NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch);
			v[6] = NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch);
			v[7] = NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch);
			v[8] = NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch);

			for (i = 0; i < 9; ++i) {
				if (min_v > v[i]) {
					min_v = v[i];
				}
			}
			NV_MAT3D_V(dest, row, col, dch) = min_v;
		}
	}
	
	for (row = 0; row < KERNEL3X3_OFFSET; ++row) {
		int col;
		float min_v;
		for (col = 0; col < dest->cols; ++col) {
			min_v = FLT_MAX;
			if (col != 0) {
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
			}
			min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			if (col != dest->cols -1) {
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			if (col != 0) {			
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
			}
			min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			if (col != dest->cols -1) {
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			NV_MAT3D_V(dest, row, col, dch) = min_v;
		}
	}
	for (row = src->rows - KERNEL3X3_OFFSET; row < dest->rows; ++row) {
		int col;
		float min_v;
		for (col = 0; col < dest->cols; ++col) {
			min_v = FLT_MAX;
			if (col != 0) {
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
			}
			min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			if (col != dest->cols -1) {
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			if (col != 0) {			
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
			}
			min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			if (col != dest->cols -1) {
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			NV_MAT3D_V(dest, row, col, dch) = min_v;
		}
	}
	for (row = 0; row < dest->rows; ++row) {
		int col;
		float min_v;
		for (col = 0; col < KERNEL3X3_OFFSET; ++col) {
			min_v = FLT_MAX;
			if (row != 0) {
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			if (row != dest->rows - 1) {
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			NV_MAT3D_V(dest, row, col, dch) = min_v;
		}
	}
	for (row = 0; row < dest->rows; ++row) {
		int col;
		float min_v;
		for (col = src->cols - KERNEL3X3_OFFSET; col < src->cols; ++col) {
			min_v = FLT_MAX;
			if (row != 0) {
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			}
			min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
			min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			if (row != dest->rows - 1) {
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
				min_v = NV_MIN(min_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			}
			NV_MAT3D_V(dest, row, col, dch) = min_v;
		}
	}
}

void
nv_dilate(nv_matrix_t *dest, int dch, const nv_matrix_t *src, int sch)
{
	int row;
	
	NV_ASSERT(dest->rows == src->rows && dest->cols == dest->cols);
#ifdef _OPENMP
#pragma omp parallel for
#endif	
	for (row = 1; row < dest->rows - 1; ++row) {
		int col;
		for (col = 1; col < dest->cols - 1; ++col) {
			int i;
			float max_v = -FLT_MAX;
			float v[9];
			
			v[0] = NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch);
			v[1] = NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch);
			v[2] = NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch);
			v[3] = NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch);
			v[4] = NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch);
			v[5] = NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch);
			v[6] = NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch);
			v[7] = NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch);
			v[8] = NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch);

			for (i = 0; i < 9; ++i) {
				if (max_v < v[i]) {
					max_v = v[i];
				}
			}
			NV_MAT3D_V(dest, row, col, dch) = max_v;
		}
	}
	
	for (row = 0; row < KERNEL3X3_OFFSET; ++row) {
		int col;
		float max_v;
		for (col = 0; col < dest->cols; ++col) {
			max_v = -FLT_MAX;
			if (col != 0) {
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
			}
			max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			if (col != dest->cols -1) {
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			if (col != 0) {			
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
			}
			max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			if (col != dest->cols -1) {
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			NV_MAT3D_V(dest, row, col, dch) = max_v;
		}
	}
	for (row = src->rows - KERNEL3X3_OFFSET; row < dest->rows; ++row) {
		int col;
		float max_v;
		for (col = 0; col < dest->cols; ++col) {
			max_v = -FLT_MAX;
			if (col != 0) {
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
			}
			max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			if (col != dest->cols -1) {
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			if (col != 0) {			
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
			}
			max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			if (col != dest->cols -1) {
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			NV_MAT3D_V(dest, row, col, dch) = max_v;
		}
	}
	for (row = 0; row < dest->rows; ++row) {
		int col;
		float max_v;
		for (col = 0; col < KERNEL3X3_OFFSET; ++col) {
			max_v = -FLT_MAX;
			if (row != 0) {
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			if (row != dest->rows - 1) {
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 2 - KERNEL3X3_OFFSET, sch));
			}
			NV_MAT3D_V(dest, row, col, dch) = max_v;
		}
	}
	for (row = 0; row < dest->rows; ++row) {
		int col;
		float max_v;
		for (col = src->cols - KERNEL3X3_OFFSET; col < src->cols; ++col) {
			max_v = -FLT_MAX;
			if (row != 0) {
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 0 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			}
			max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
			max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 1 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			if (row != dest->rows - 1) {
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 0 - KERNEL3X3_OFFSET, sch));
				max_v = NV_MAX(max_v,  NV_MAT3D_V(src, row + 2 - KERNEL3X3_OFFSET, col + 1 - KERNEL3X3_OFFSET, sch));
			}
			NV_MAT3D_V(dest, row, col, dch) = max_v;
		}
	}
}

void
nv_morph_open(nv_matrix_t *dest, int dch, const nv_matrix_t *src, int sch)
{
	nv_matrix_t *tmp = nv_matrix3d_alloc(1, dest->rows, dest->cols);
	
	NV_ASSERT(dest->rows == src->rows && dest->cols == src->cols);
	
	nv_erode(tmp, 0, src, sch);
	nv_dilate(dest, dch, tmp ,0);
	
	nv_matrix_free(&tmp);
}

void
nv_morph_close(nv_matrix_t *dest, int dch, const nv_matrix_t *src, int sch)
{
	nv_matrix_t *tmp = nv_matrix3d_alloc(1, dest->rows, dest->cols);
	
	NV_ASSERT(dest->rows == src->rows && dest->cols == src->cols);
		
	nv_dilate(tmp, 0, src, sch);
	nv_erode(dest, dch, tmp, 0);
	
	nv_matrix_free(&tmp);
}
