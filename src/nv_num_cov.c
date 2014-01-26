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
#include "nv_num_vector.h"
#include "nv_num_cov.h"
#include "nv_num_matrix.h"
#include "nv_num_eigen.h"

#define NV_COV_EIGEN_MAX_EPOCH 100

/* 分散共分散 */

nv_cov_t *nv_cov_alloc(int n)
{
	nv_cov_t *cov = (nv_cov_t *)nv_malloc(sizeof(nv_cov_t));

	cov->n = n;
	cov->u = nv_matrix_alloc(n, 1);
	cov->cov = nv_matrix_alloc(n, n);
	cov->eigen_val = nv_matrix_alloc(1, n);
	cov->eigen_vec = nv_matrix_alloc(n, n);
	cov->data_m = 0;

	return cov;
}

void nv_cov_free(nv_cov_t **cov)
{
	if (*cov) {
		nv_matrix_free(&(*cov)->cov);
		nv_matrix_free(&(*cov)->eigen_val);
		nv_matrix_free(&(*cov)->eigen_vec);
		nv_matrix_free(&(*cov)->u);

		nv_free(*cov);
		*cov = NULL;
	}
}

void
nv_cov_eigen_ex(nv_cov_t *cov, const nv_matrix_t *data,
				int eigen_n, int max_epoch)
{
	nv_cov(cov->cov, cov->u, data);
    /* 高速化のためべき乗法を使うのであまり大きなeigen_nを指定しないように注意 */
	nv_eigen(cov->eigen_vec, cov->eigen_val, cov->cov, eigen_n, max_epoch);
	cov->data_m = data->m;
}

void
nv_cov_eigen(nv_cov_t *cov, const nv_matrix_t *data)
{
	nv_cov(cov->cov, cov->u, data);
	nv_eigen_sym(cov->eigen_vec, cov->eigen_val, cov->cov,
				 NV_COV_EIGEN_MAX_EPOCH);
	cov->data_m = data->m;
}

void nv_cov(nv_matrix_t *cov,
			nv_matrix_t *u,
			const nv_matrix_t *data)
{
	int m;
	int alloc_u = 0;
	const float factor = (data->m > 1) ? 1.0f / (data->m - 1): 1.0f / data->m;
	
	if (u == NULL) {
		u = nv_matrix_alloc(cov->n, 1);
		alloc_u =1;
	}
	NV_ASSERT(cov->n == data->n && cov->n == cov->m
			  && u->n == cov->n);

	/* 平均 */
	nv_matrix_mean(u, 0, data);

	/* 分散共分散行列 */
	nv_matrix_zero(cov);

	for (m = 0; m < cov->m; ++m) {
		nv_matrix_t *dum = nv_matrix_alloc(data->m, 1);
		int j, n;
		const float um = NV_MAT_V(u, 0, m);
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (j = 0; j < data->m; ++j) {
			NV_MAT_V(dum, 0, j) = (NV_MAT_V(data, j, m) - um) * factor;
		}
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
		for (n = m; n < cov->n; ++n) {
			float v = 0.0f;
			int i;
			const float un = NV_MAT_V(u, 0, n);
			for (i = 0; i < data->m; ++i) {
				v += (NV_MAT_V(data, i, n) - un) * NV_MAT_V(dum, 0, i);
			}
			NV_MAT_V(cov, m, n) = NV_MAT_V(cov, n, m) = v;
		}
		nv_matrix_free(&dum);
	}
	if (alloc_u) {
		nv_matrix_free(&u);
	}
}
