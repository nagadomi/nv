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
#include "nv_num_cov.h"
#include "nv_num_matrix.h"
#include "nv_num_eigen.h"

#define NV_COV_JACOBI_EPOCH 100

/* 分散共分散 */

nv_cov_t *nv_cov_alloc(int n)
{
	nv_cov_t *cov = (nv_cov_t *)nv_malloc(sizeof(nv_cov_t));

	cov->n = n;
	cov->u = nv_matrix_alloc(n, 1);
	cov->s = nv_matrix_alloc(n, 1);
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
		nv_matrix_free(&(*cov)->s);
		nv_matrix_free(&(*cov)->u);

		nv_free(*cov);
		*cov = NULL;
	}
}

void
nv_cov_eigen_ex(nv_cov_t *cov, const nv_matrix_t *data,
				int max_epoch)
{
	nv_cov(cov->cov, cov->u, cov->s, data);
	nv_eigen_sym(cov->eigen_vec, cov->eigen_val,
				 cov->cov, max_epoch);
	cov->data_m = data->m;
}

void
nv_cov_eigen(nv_cov_t *cov, const nv_matrix_t *data)
{
	nv_cov(cov->cov, cov->u, cov->s, data);
	nv_eigen_sym(cov->eigen_vec, cov->eigen_val,
						cov->cov, NV_COV_JACOBI_EPOCH);
	cov->data_m = data->m;
}

void nv_cov(nv_matrix_t *cov,
			nv_matrix_t *u,
			nv_matrix_t *s,
			const nv_matrix_t *data)
{
	int m, n;
	int alloc_u = 0;
	int alloc_s = 0;
	float factor = 1.0f / data->m;

	if (u == NULL) {
		u = nv_matrix_alloc(cov->n, 1);
		alloc_u =1;
	}
	if (s == NULL) {
		s = nv_matrix_alloc(cov->n, 1);
		alloc_s =1;
	}
	NV_ASSERT(cov->n == data->n && cov->n == cov->m
		&& u->n == cov->n
		&& s->n == cov->n);

	/* 平均 */
	nv_matrix_zero(u);
	for (m = 0; m < data->m; ++m) {
		for (n = 0; n < data->n; ++n) {
			NV_MAT_V(u, 0, n) += NV_MAT_V(data, m, n) * factor;
		}
	}

	/* 分散共分散行列 */
	nv_matrix_zero(cov);
	nv_matrix_zero(s);

	for (m = 0; m < cov->m; ++m) {
		nv_matrix_t *dum = nv_matrix_alloc(data->m, 1);
		int i;
		float um = NV_MAT_V(u, 0, m);
		for (i = 0; i < data->m; ++i) {
			NV_MAT_V(dum, 0, i) = (NV_MAT_V(data, i, m) - um) * factor;
		}
		for (n = m; n < cov->n; ++n) {
			float v = 0.0f;
			float un = NV_MAT_V(u, 0, n);
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
	if (alloc_s) {
		nv_matrix_free(&s);
	}
}
