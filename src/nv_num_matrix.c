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
	nv_matrix_t *tr = nv_matrix_alloc(mat->m, mat->n);
	nv_matrix_tr_ex(tr, mat);
	return tr;
}

void
nv_matrix_tr_ex(nv_matrix_t *tr, const nv_matrix_t *mat)
{
	int j, i;
	NV_ASSERT(tr->n == mat->m);
	NV_ASSERT(tr->m == mat->n);
	for (j = 0; j < mat->m; ++j) {
		for (i = 0; i < mat->n; ++i) {
			NV_MAT_V(tr, i, j) = NV_MAT_V(mat, j, i);
		}
	}
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
			nv_vector_zero(y, yj);
			for (i = 0; i < a->m; ++i) {
				int j;
				for (j = 0; j < a->n; ++j) {
					NV_MAT_V(y, yj, i) += NV_MAT_V(a, j, i) * NV_MAT_V(x, xj, j);
				}
			}
		}
	}
}

/* TODO: できるだけ転置せずにベクトル命令で効率化 */
static void
matrix_mul(nv_matrix_t *y,
		   const nv_matrix_t *a,
		   const nv_matrix_t *b)
{
    int m = a->m;
    int n = a->n;
    int p = b->n;
	int i, j, k;
	
	NV_ASSERT(a->n == b->m);
	NV_ASSERT(y->n == a->n);
	NV_ASSERT(y->m == a->m);
	NV_ASSERT(y->n == b->n);

	nv_matrix_zero(y);
	
	for(i = 0; i < m; i++){
		for(j = 0; j < p; j++){
			for(k = 0; k < n; k++){
				NV_MAT_V(y, i, j) += NV_MAT_V(a, i, k) * NV_MAT_V(b, k, j);
			}
		}
	}
}

void
nv_matrix_mul(nv_matrix_t *y,
			  const nv_matrix_t *a_,
			  nv_matrix_tr_t a_tr,
			  const nv_matrix_t *b_,
			  nv_matrix_tr_t b_tr)
{
	if ((a_tr == NV_MAT_TR && b_tr == NV_MAT_TR)) {
		nv_matrix_t *a = nv_matrix_tr(a_);
		nv_matrix_t *b = nv_matrix_tr(b_);
		matrix_mul(y, a, b);
		nv_matrix_free(&a);
		nv_matrix_free(&b);
	} else if (a_tr == NV_MAT_NOTR && b_tr == NV_MAT_NOTR) {
		const nv_matrix_t *a = a_;
		const nv_matrix_t *b = b_;
		matrix_mul(y, a, b);
	} else if (a_tr == NV_MAT_NOTR && b_tr == NV_MAT_TR) {
		const nv_matrix_t *a = a_;
		nv_matrix_t *b = nv_matrix_tr(b_);
		matrix_mul(y, a, b);
		nv_matrix_free(&b);
	} else if (a_tr == NV_MAT_TR && b_tr == NV_MAT_NOTR) {
		nv_matrix_t *a = nv_matrix_tr(a_);
		const nv_matrix_t *b = b_;
		matrix_mul(y, a, b);
		nv_matrix_free(&a);
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
nv_matrix_adds(nv_matrix_t *y, const nv_matrix_t *a, float val)
{
	int i;
	
	NV_ASSERT(y->m == a->m);
	NV_ASSERT(y->n == a->n);
	
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (i = 0; i < y->m; ++i) {
		nv_vector_adds(y, i, a, i, val);
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
nv_matrix_sub(nv_matrix_t *y, const nv_matrix_t *a, const nv_matrix_t *b)
{
	int i;
	NV_ASSERT(y->m == a->m && y->m == b->m);
	NV_ASSERT(y->n == a->n && y->n == b->n);
	
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (i = 0; i < y->m; ++i) {
		nv_vector_sub(y, i, a, i, b, i);
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

void
nv_matrix_mean(nv_matrix_t *y, int y_j, const nv_matrix_t *data)
{
	int procs = nv_omp_procs();
	float scale = 1.0f / data->m;
	nv_matrix_t *tmp = nv_matrix_alloc(data->n, procs);
	nv_matrix_t *scale_vec = nv_matrix_alloc(data->n, procs);
	int j, i;

	NV_ASSERT(data->n == y->n);	
	
	nv_matrix_zero(tmp);
#ifdef _OPENMP
#pragma omp parallel for num_threads(procs)
#endif
	for (j = 0; j < data->m; ++j) {
		int thread_id = nv_omp_thread_id();
		nv_vector_muls(scale_vec, thread_id, data, j, scale);
		nv_vector_add(tmp, thread_id, tmp, thread_id, scale_vec, thread_id);
	}
	nv_vector_zero(y, y_j);
	for (i = 0; i < procs; ++i) {
		nv_vector_add(y, y_j, y, y_j, tmp, i);
	}
	
	nv_matrix_free(&tmp);
	nv_matrix_free(&scale_vec);
}
	
void
nv_matrix_var(nv_matrix_t *y, int y_j,
			  const nv_matrix_t *data)
{
	nv_matrix_t *mean = nv_matrix_alloc(data->n, 1);
	
	NV_ASSERT(data->n == y->n);
	
	nv_matrix_mean(mean, 0, data);
	nv_matrix_var_ex(y, y_j, data, mean, 0);
	nv_matrix_free(&mean);
}

void
nv_matrix_var_ex(nv_matrix_t *y, int y_j,
				 const nv_matrix_t *data,
				 const nv_matrix_t *mean,
				 int mean_j)
{
	int procs = nv_omp_procs();
	nv_matrix_t *tmp = nv_matrix_alloc(data->n, procs);
	nv_matrix_t *sub = nv_matrix_alloc(data->n, procs);
	float scale = (data->m > 1) ? (1.0f / (data->m - 1)) : data->m;
	int i, j;
	
	NV_ASSERT(data->n == y->n);
	NV_ASSERT(mean->n == y->n);	
	
	nv_matrix_zero(tmp);
	nv_vector_zero(y, y_j);
	
#ifdef _OPENMP
#pragma omp parallel for num_threads(procs)
#endif		
	for (j = 0; j < data->m; ++j) {
		int thread_id = nv_omp_thread_id();
		nv_vector_sub(sub, thread_id, mean, mean_j, data, j);
		nv_vector_mul(sub, thread_id, sub, thread_id, sub, thread_id);
		nv_vector_muls(sub, thread_id, sub, thread_id, scale);
		nv_vector_add(tmp, thread_id, tmp, thread_id, sub, thread_id);
	}
	for (i = 0; i < procs; ++i) {
		nv_vector_add(y, y_j, y, y_j, tmp, i);
	}
	
	nv_matrix_free(&tmp);
	nv_matrix_free(&sub);
}

void
nv_matrix_normalize_shift(nv_matrix_t *mat, float min_v, float max_v)
{
	float mat_min = FLT_MAX;
	float mat_max = -FLT_MAX;
	int i;
	float scale;
	
	for (i = 0; i < mat->m; ++i) {
		float vec_max = nv_vector_maxs(mat, i);
		float vec_min = nv_vector_mins(mat, i);
		if (mat_max < vec_max) {
			mat_max = vec_max;
		}
		if (mat_min > vec_min) {
			mat_min = vec_min;
		}
	}
	scale = (max_v - min_v) / (mat_max - mat_min);
	for (i = 0; i < mat->m; ++i) {
		nv_vector_subs(mat, i, mat, i, mat_min);
		nv_vector_muls(mat, i, mat, i, scale);
		nv_vector_adds(mat, i, mat, i, min_v);
	}
}
