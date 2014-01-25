/*
 * This file is part of libnv.
 *
 * Copyright (C) 2014 nagadomi@nurs.or.jp
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
#include "nv_num_matrix.h"
#include "nv_num_cov.h"
#include "nv_num_eigen.h"
#include "nv_num_zca.h"

/* ZCA Whitening */

void
nv_zca_train(nv_matrix_t *mean, int mean_j,
			 nv_matrix_t *u,
			 const nv_matrix_t *a,
			 float epsilon)
{
	nv_zca_train_ex(mean, mean_j, u, a, epsilon, a->n, 1);
}

void
nv_zca_train_ex(nv_matrix_t *mean, int mean_j,
				nv_matrix_t *u,
				const nv_matrix_t *a,
				float epsilon,
				int npca,
				int keep_space)
{
	nv_cov_t *cov = nv_cov_alloc(a->n);
	nv_matrix_t *d = nv_matrix_alloc(a->n, a->n);
	nv_matrix_t *v = nv_matrix_alloc(a->n, a->n);
	nv_matrix_t *eig;
	int i;

	// [V,D] = eig(cov(A))
	// U = V * diag(sqrt(1.0f / (diag(D) + epsilon))) * V'
	nv_matrix_zero(d);
	nv_cov_eigen(cov, a);
	for (i = 0; i < d->n; ++i) {
		if (i < npca) {
			NV_MAT_V(d, i, i) = sqrtf(1.0f / (NV_MAT_V(cov->eigen_val, i, 0) + epsilon));
		}
	}
	eig = nv_matrix_tr(cov->eigen_vec);
	nv_matrix_mul(v, eig, NV_MAT_NOTR, d, NV_MAT_NOTR);
	if (keep_space) {
		nv_matrix_t *tr;
		nv_matrix_mul(u, v, NV_MAT_NOTR, eig, NV_MAT_TR);
		tr = nv_matrix_tr(u);
		nv_matrix_copy_all(u, tr);
		nv_matrix_free(&tr);
	} else {
		nv_matrix_t *tr;
		tr = nv_matrix_tr(v);
		nv_matrix_copy_all(u, tr);
		nv_matrix_free(&tr);
	}
	nv_vector_copy(mean, mean_j, cov->u, 0);
	
	nv_matrix_free(&eig);
	nv_matrix_free(&d);
	nv_matrix_free(&v);
}

void
nv_zca_whitening(nv_matrix_t *x, int x_j,
				 const nv_matrix_t *mean, int mean_j,
				 const nv_matrix_t *u)
{
	nv_matrix_t *sub = nv_matrix_alloc(x->n, 1);
	
	// ZCAWhitend_x = (x - mean) * U
	nv_vector_sub(sub, 0, x, x_j, mean, mean_j);
	nv_vector_mulmtr(x, x_j, sub, 0, u);
	nv_matrix_free(&sub);
}

void
nv_zca_whitening_all(nv_matrix_t *a,
					 const nv_matrix_t *mean, int mean_j,
					 const nv_matrix_t *u)
{
	int j;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (j = 0; j < a->m; ++j) {
		nv_zca_whitening(a, j, mean, mean_j, u);
	}
}
