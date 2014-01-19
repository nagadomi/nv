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
#include "nv_num_matrix.h"
#include "nv_num_vector.h"

nv_matrix_t *
nv_matrix_tr(const nv_matrix_t *mat)
{
	int j, i;
	nv_matrix_t *tr = nv_matrix_alloc(mat->m, mat->n);
	for (j = 0; j < mat->m; ++j) {
		for (i = 0; i < mat->n; ++i) {
			NV_MAT_V(tr, i, j) = NV_MAT_V(mat, j, i);
		}
	}

	return tr;
}

nv_matrix_t *
nv_matrix3d_tr(const nv_matrix_t *mat)
{
	int row, col, i;
	nv_matrix_t *tr = nv_matrix3d_alloc(mat->n, mat->cols, mat->rows);

	for (row = 0; row < mat->rows; ++row) {
		for (col = 0; col < mat->cols; ++col) {
			for (i = 0; i < mat->n; ++i) {
				NV_MAT3D_V(tr, col, row, i) = NV_MAT3D_V(mat, row, col, i);
			}
		}
	}

	return tr;
}

void
nv_matrix_mulv(nv_matrix_t *y, int yj,
			   const nv_matrix_t *a,
			   nv_matrix_tr_t a_tr,
			   const nv_matrix_t *x,
			   int xj)
{
	int i;

	if (a_tr == NV_MAT_TR) {
		NV_ASSERT(x->n == a->n);
		NV_ASSERT(y->n == a->m);
		for (i = 0; i < a->m; ++i) {
			NV_MAT_V(y, yj, i) = nv_vector_dot(a, i, x, xj);
		}
	} else {
		NV_ASSERT(x->n == a->m);
		NV_ASSERT(y->n == a->n);
		if (a->n > 256 || a->m > 256) {
			nv_matrix_t *a2 = nv_matrix_tr(a);
			for (i = 0; i < a2->m; ++i) {
				NV_MAT_V(y, yj, i) = nv_vector_dot(a2, i, x, xj);
			}
			nv_matrix_free(&a2);
		} else {
			for (i = 0; i < a->m; ++i) {
				int j;
				for (j = 0; j < a->n; ++j) {
					NV_MAT_V(y, yj, i) += NV_MAT_V(a, j, i) * NV_MAT_V(x, xj, j);
				}
			}
		}
	}
}

void nv_matrix_mul(nv_matrix_t *y,
				   const nv_matrix_t *a_,
				   nv_matrix_tr_t a_tr,
				   const nv_matrix_t *b_,
				   nv_matrix_tr_t b_tr)
{
	int i;
	if ((a_tr == NV_MAT_TR && b_tr == NV_MAT_TR)) {
		const nv_matrix_t *a = a_;
		nv_matrix_t *b = nv_matrix_tr(b_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (i = 0; i < y->m; ++i) {
			nv_matrix_mulv(y, i, a, NV_MAT_TR, b, i);
		}
		nv_matrix_free(&b);
	} else if (a_tr == NV_MAT_NOTR && b_tr == NV_MAT_NOTR) {
		nv_matrix_t *a = nv_matrix_tr(a_);
		const nv_matrix_t *b = b_;
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (i = 0; i < y->m; ++i) {
			nv_matrix_mulv(y, i, a, NV_MAT_TR, b, i);
		}
		nv_matrix_free(&a);
	} else if (a_tr == NV_MAT_NOTR && b_tr == NV_MAT_TR) {
		const nv_matrix_t *a = a_;
		const nv_matrix_t *b = b_;
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (i = 0; i < y->m; ++i) {
			nv_matrix_mulv(y, i, a, NV_MAT_TR, b, i);
		}
	} else if (a_tr == NV_MAT_TR && b_tr == NV_MAT_NOTR) {
		nv_matrix_t *a = nv_matrix_tr(a_);
		nv_matrix_t *b = nv_matrix_tr(a_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (i = 0; i < y->m; ++i) {
			nv_matrix_mulv(y, i, a, NV_MAT_TR, b, i);
		}
		nv_matrix_free(&a);
		nv_matrix_free(&b);		
	}
}

void
nv_matrix_muls(nv_matrix_t *y, const nv_matrix_t *a, float scale)
{
	int i;
	
	NV_ASSERT(y->m == a->m);
	NV_ASSERT(y->n == a->n);
	
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (i = 0; i < y->m; ++i) {
		nv_vector_muls(y, i, a, i, scale);
	}
}

void
nv_matrix_add(nv_matrix_t *y, const nv_matrix_t *a, const nv_matrix_t *b)
{
	int i;
	NV_ASSERT(y->m == a->m && y->m == b->m);
	NV_ASSERT(y->n == a->n && y->n == b->n);
	
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (i = 0; i < y->m; ++i) {
		nv_vector_add(y, i, a, i, b, i);
	}
}

void
nv_matrix_diag(nv_matrix_t *diag,
					nv_matrix_t *vec,
					int vec_j)
{
	int i;
	
	NV_ASSERT(diag->n == diag->m);
	NV_ASSERT(diag->n == vec->n);
	
	nv_matrix_zero(diag);
	for (i = 0; i < diag->m; ++i) {
		NV_MAT_V(diag, i, i) = NV_MAT_V(vec, vec_j, i);
	}
}
