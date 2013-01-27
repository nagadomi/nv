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
#if 0
#include "nv_num_lapack.h"
#endif

/* 転置 */
nv_matrix_t *nv_matrix_tr(const nv_matrix_t *mat)
{
	int m, n;
	nv_matrix_t *tr = nv_matrix_alloc(mat->m, mat->n);

	for (m = 0; m < mat->m; ++m) {
		for (n = 0; n < mat->n; ++n) {
			NV_MAT_V(tr, n, m) = NV_MAT_V(mat, m, n);		
		}
	}

	return tr;
}

nv_matrix_t *nv_matrix3d_tr(const nv_matrix_t *mat)
{
	int row, col, n;
	nv_matrix_t *tr = nv_matrix3d_alloc(mat->n, mat->cols, mat->rows);

	for (row = 0; row < mat->rows; ++row) {
		for (col = 0; col < mat->cols; ++col) {
			for (n = 0; n < mat->n; ++n) {
				NV_MAT3D_V(tr, col, row, n) = NV_MAT3D_V(mat, row, col, n);
			}
		}
	}

	return tr;
}

void nv_gemv(nv_matrix_t *y, int ym,
			 nv_matrix_tr_t a_tr,
			 const nv_matrix_t *a,
			 const nv_matrix_t *x,
			 int xm)
{
	int m;

	if (a_tr == NV_MAT_TR) {
		NV_ASSERT(x->n == a->n);
		NV_ASSERT(y->n == a->m);

#ifdef _OPENMP
#pragma omp parallel for if (a->m > 512)
#endif
		for (m = 0; m < a->m; ++m) {
			NV_MAT_V(y, ym, m) = nv_vector_dot(a, m, x, xm);
		}
	} else {
		NV_ASSERT(x->n == a->n);
		NV_ASSERT(y->n == a->m);

		nv_vector_zero(y, ym);

#ifdef _OPENMP
#pragma omp parallel for if (a->m > 512)
#endif
		for (m = 0; m < a->m; ++m) {
			int n;
			for (n = 0; n < a->n; ++n) {
				NV_MAT_V(y, ym, m) += NV_MAT_V(a, n, m) * NV_MAT_V(x, xm, n);;
			}
		}
	}
}

#if 0
/* 連立一次方程式 特異値分解+近似 */
int nv_gelss(nv_matrix_t *x,
			 nv_matrix_t *s, /* NxN Matrix */
			 const nv_matrix_t *a, /* NxN Matrix */
			 const nv_matrix_t *b)
{
	nv_matrix_t *t_a = nv_matrix_alloc(a->n, a->m);
	nv_matrix_t *t_b = nv_matrix_alloc(b->n, b->m);
	nv_matrix_t *t_sv = nv_matrix_alloc(a->n, 1);
	real epsilon = FLT_EPSILON;
	integer rank = a->n;
	integer lwork = 3 * b->n + 2 * b->n;
	real *work = (real *)nv_malloc(sizeof(real) * lwork);
	integer t_a_m = (integer)t_a->m;
	integer t_b_m = (integer)t_b->m;
	integer t_a_n = (integer)t_a->n;
	integer t_a_step = (integer)t_a->step;
	integer t_b_step = (integer)t_b->step;
	integer nrhs = 1;
	integer info = 0;

	NV_ASSERT(a->n == a->m);
	NV_ASSERT(a->n == b->n);
	NV_ASSERT(x->n == b->n);
	NV_ASSERT(x->m == b->m);

	nv_matrix_copy(t_a, 0, a, 0, a->m);
	nv_matrix_copy(t_b, 0, b, 0, b->m);
	nv_matrix_zero(t_sv);

	sgelss_(&t_a_m, &t_a_n, &t_b_m, t_a->v, &t_a->step, t_b->v, &t_b_step, t_sv->v, &epsilon, &rank, work, &lwork, &info);
	nv_matrix_copy(x, 0, t_b, 0, x->m);
	if (s != NULL) {
		nv_matrix_copy(s, 0, t_sv, 0, s->m);
	}
	nv_matrix_free(&t_a);
	nv_matrix_free(&t_b);
	nv_matrix_free(&t_sv);
	nv_free(work);

	return info;
}
#endif


void nv_matrix_normalize_maxmin(nv_matrix_t *mat, int mat_n, float min_v, float max_v)
{
	int m;
	float cur_max_v = -FLT_MAX;
	float cur_min_v = FLT_MAX;

	if (mat_n >= 0) {
		for (m = 0; m < mat->m; ++m) {
			if (NV_MAT_V(mat, m, mat_n) > cur_max_v) {
				cur_max_v = NV_MAT_V(mat, m, mat_n);
			}
			if (NV_MAT_V(mat, m, mat_n) < cur_min_v) {
				cur_min_v = NV_MAT_V(mat, m, mat_n);
			}
		}
			
		if (fabsf(cur_max_v - cur_min_v) > FLT_EPSILON) {
			float scale = (max_v - min_v) / (cur_max_v - cur_min_v);
			for (m = 0; m < mat->m; ++m) {
				NV_MAT_V(mat, m, mat_n) = (NV_MAT_V(mat, m, mat_n) - cur_min_v) * scale + min_v;
				if (NV_MAT_V(mat, m, mat_n) > max_v) {
					NV_MAT_V(mat, m, mat_n) = max_v;
				}
				if (NV_MAT_V(mat, m, mat_n) < min_v) {
					NV_MAT_V(mat, m, mat_n) = min_v;
				}
			}
		}
	} else {
		int n;
		for (m = 0; m < mat->m; ++m) {
			for (n = 0; n < mat->n; ++n) {
				if (NV_MAT_V(mat, m, n) > cur_max_v) {
					cur_max_v = NV_MAT_V(mat, m, n);
				}
				if (NV_MAT_V(mat, m, n) < cur_min_v) {
					cur_min_v = NV_MAT_V(mat, m, n);
				}
			}
		}
		if (fabsf(cur_max_v - cur_min_v) > FLT_EPSILON) {
			float scale = (max_v - min_v) / (cur_max_v - cur_min_v);
			for (m = 0; m < mat->m; ++m) {
				for (n = 0; n < mat->n; ++n) {
					NV_MAT_V(mat, m, n) = (NV_MAT_V(mat, m, n) - cur_min_v) * scale + min_v;
					if (NV_MAT_V(mat, m, n) > max_v) {
						NV_MAT_V(mat, m, n) = max_v;
					}
					if (NV_MAT_V(mat, m, n) < min_v) {
						NV_MAT_V(mat, m, n) = min_v;
					}
				}
			}
		}
	}
}

void
nv_matrix_muls(nv_matrix_t *mat, float scale)
{
	int i;
#ifdef _OPENMP
#pragma omp parallel for	
#endif
	for (i = 0; i < mat->m; ++i) {
		nv_vector_muls(mat, i, mat, i, scale);
	}
}
