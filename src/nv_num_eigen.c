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
#include "nv_num_eigen.h"

/* べき乗法で上位Nの固有値,固有ベクトルを求める */

int
nv_eigen(nv_matrix_t *eigen_vec, 
		 nv_matrix_t *eigen_val,
		 const nv_matrix_t *mat,
		 int n,
		 int max_epoch)
{
	int i;
	nv_matrix_t *a = nv_matrix_dup(mat);
	nv_matrix_t *vec_tmp = nv_matrix_alloc(a->m, 1);
#if NV_ENABLE_SSE2	
	const int pk_lp = (a->n & 0xfffffffc);
#endif
	
	NV_ASSERT(n > 0);
	NV_ASSERT(n <= mat->m);
	NV_ASSERT(n <= eigen_vec->m);
	NV_ASSERT(n <= eigen_val->m);
	NV_ASSERT(mat->m == mat->n);
	NV_ASSERT(mat->m == eigen_vec->n);

	nv_matrix_zero(eigen_val);
	nv_matrix_fill(eigen_vec, 1.0f);
	nv_vector_normalize_all(eigen_vec);
	
	for (i = 0; i < n; ++i) {
		int k, jj;
		float lambda_old;
		
		for (k = 0; k < max_epoch; ++k) {
			int j;
			float lambda;
			
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (j = 0; j < a->m; ++j) {
				NV_MAT_V(vec_tmp, 0, j) = nv_vector_dot(a, j, eigen_vec, i);
			}
			lambda = nv_vector_norm(vec_tmp, 0);
			if (lambda > 0.0f) {
				nv_vector_muls(vec_tmp, 0, vec_tmp, 0, 1.0f / lambda);
			}
			NV_MAT_V(eigen_val, i, 0) = lambda;
			nv_vector_copy(eigen_vec, i, vec_tmp, 0);
			
			if (k > 0) {
				if (fabsf(lambda_old - lambda) < FLT_EPSILON) {
					break;
				}
			}
			lambda_old = NV_MAT_V(eigen_val, i, 0);
		}
#if NV_ENABLE_SSE2
		{
			const __m128 val = _mm_set1_ps(NV_MAT_V(eigen_val, i, 0));
			
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (jj = 0; jj < a->m; ++jj) {
				int ii;
				const __m128 vjj = _mm_set1_ps(NV_MAT_V(eigen_vec, i, jj));
				for (ii = 0; ii < pk_lp; ii += 4) {
					_mm_store_ps(&NV_MAT_V(a, jj, ii),
								 _mm_sub_ps(*(const __m128 *)&NV_MAT_V(a, jj, ii),
											_mm_mul_ps(val,_mm_mul_ps(vjj, *(const __m128 *)&NV_MAT_V(eigen_vec, i, ii)))));
				}
				for (; ii < a->n; ++ii) {
					NV_MAT_V(a, jj, ii) -=
						NV_MAT_V(eigen_val, i, 0)
						* NV_MAT_V(eigen_vec, i, ii)
						* NV_MAT_V(eigen_vec, i, jj);
				}
			}
		}
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (jj = 0; jj < a->m; ++jj) {
			int ii;
			for (ii = 0; ii < a->n; ++ii) {
				NV_MAT_V(a, jj, ii) -=
					NV_MAT_V(eigen_val, i, 0)
					* NV_MAT_V(eigen_vec, i, ii)
					* NV_MAT_V(eigen_vec, i, jj);
			}
		}
#endif		
	}
	nv_matrix_free(&vec_tmp);
	nv_matrix_free(&a);
	
	return 0;
}
