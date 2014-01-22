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

#define NV_EIGEN_EQ(a, b) (fabsf(a - b) <= FLT_EPSILON * NV_MAX(fabsf(a), fabsf(b)))

/* べき乗法で上位Nの固有値,固有ベクトルを求める
 *
 * 固有値が小さい方は誤差が伝搬して信頼できないので
 * 全固有値を使う場合はこの関数を使うべきではない
 */
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
		float lambda_old = 0.0f;
		
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
				nv_vector_divs(vec_tmp, 0, vec_tmp, 0, lambda);
			}
			NV_MAT_V(eigen_val, i, 0) = lambda;
			nv_vector_copy(eigen_vec, i, vec_tmp, 0);
			nv_vector_normalize_L2(eigen_vec, i);
			if (k > 0) {
				if (NV_EIGEN_EQ(lambda_old, lambda)) {
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

/* 巡回Jacobi法で実対称行列の固有値, 固有ベクトルを求める.
 * 参考: Numerical Recipes in C
 */
static int 
sort_eigen(const void *p1, const void *p2)
{
        float *f1 = (float *)p1;
        float *f2 = (float *)p2;

        if (f1[0] == f2[0]) {
                return 0;
        }

        return f1[0] < f2[0] ? 1:-1;
}
int 
nv_eigen_sym(nv_matrix_t *eigen_vec, 
			 nv_matrix_t *eigen_val,
			 const nv_matrix_t *smat,
			 int max_epoch)
{
	int i, converge;
	const int n = smat->n;
	const float n2_inv = 1.0f / (n * n);
	nv_matrix_t *b, *z;
	nv_matrix_t *v = eigen_vec;
	nv_matrix_t *d = eigen_val;
	nv_matrix_t *a = nv_matrix_alloc(smat->n, smat->m);

	NV_ASSERT(
		smat->m == smat->n
		&& eigen_vec->n == smat->n
		&& eigen_vec->m == smat->m
		&& eigen_val->n == 1
		&& eigen_val->m == eigen_vec->m
        );

	nv_matrix_copy(a, 0, smat, 0, smat->m);
	b = nv_matrix_alloc(1, n);
	z = nv_matrix_alloc(1, n);

	nv_matrix_zero(z);
	nv_matrix_zero(v);

	/* 単位行列に初期化 */
	for (i = 0; i < n; ++i) {
		NV_MAT_V(v, i, i) = 1.0f;
	}
	/* d, bをaの対角成分で初期化 */
	for (i = 0; i < n; ++i) {
		NV_MAT_V(b, i, 0) = NV_MAT_V(d, i, 0) = NV_MAT_V(a, i, i);
	}

	converge = 0;
	for (i = 0; i < max_epoch; ++i) {
		/* 収束判定 */
		float sm = 0.0f;
		float tresh;
		int ip;
                
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sm) schedule(dynamic, 16)
#endif
		for (ip = 0; ip < n - 1; ++ip) {
			int iq;
			for (iq = ip + 1; iq < n; ++iq) {
				sm += fabsf(NV_MAT_V(a, ip, iq));
			}
		}
		if (sm <= FLT_EPSILON) {
			converge = 1;
			break;
		}
		tresh = (i > 3)  ? 0.0f: (0.2f * sm * n2_inv);
		
		/* 上三角成分(対角成分含まず)が0になるように回転していく */
		for (ip = 0; ip < n - 1; ++ip) {
			int iq;
			for (iq = ip + 1; iq < n; ++iq) {
				float g = 100.0f * fabsf(NV_MAT_V(a, ip, iq));
				if (i > 3 && g <= FLT_EPSILON) {
					NV_MAT_V(a, ip, iq) = 0.0f;
				} else if (fabs(NV_MAT_V(a, ip, iq)) > tresh) {
					float t, h, theta, c, s, tau;
					
					h = NV_MAT_V(d, iq, 0) - NV_MAT_V(d, ip, 0);
					if (g <= FLT_EPSILON) {
						t = NV_MAT_V(a, ip, iq) / h;
					} else {
						theta = 0.5f * h / NV_MAT_V(a, ip, iq);
						t = 1.0f / (fabsf(theta) + sqrtf(1.0f + theta * theta));
						if (theta < 0.0f) {
							t = -t;
						}
					}
					c = 1.0f / sqrtf(1.0f + t * t);
					s = t * c;
					tau = s / (1.0f + c);
					h = t * NV_MAT_V(a, ip, iq);
					NV_MAT_V(z, ip, 0) -= h;
					NV_MAT_V(z, iq, 0) += h;
					NV_MAT_V(d, ip, 0) -= h;
					NV_MAT_V(d, iq, 0) += h;
					NV_MAT_V(a, ip, iq) = 0.0f;
#ifdef _OPENMP
#pragma omp parallel sections
#endif
					{                                        
#ifdef _OPENMP
#pragma omp section
#endif
						{
							int j;
							for (j = 0; j < ip; ++j) {
								const float g = NV_MAT_V(a, j, ip);
								const float h = NV_MAT_V(a, j, iq);
								NV_MAT_V(a, j, ip) = g - s * (h + g * tau);
								NV_MAT_V(a, j, iq) = h + s * (g - h * tau); 
							}
							for (j = ip + 1; j < iq; ++j) {
								const float g = NV_MAT_V(a, ip, j); 
								const float h = NV_MAT_V(a, j, iq); 
								NV_MAT_V(a, j, iq) = h + s * (g - h * tau); 
								NV_MAT_V(a, ip, j) = g - s * (h + g * tau); 
							}
							for (j = iq + 1; j < n; ++j) {
								const float g = NV_MAT_V(a, ip, j); 
								const float h = NV_MAT_V(a, iq, j); 
								NV_MAT_V(a, iq, j) = h + s * (g - h * tau); 
								NV_MAT_V(a, ip, j) = g - s * (h + g * tau); 
							}
						}
#ifdef _OPENMP
#pragma omp section
#endif
						{
							int j;
							for (j = 0; j < n; ++j) {
								const float g = NV_MAT_V(v, j, ip); 
								const float h = NV_MAT_V(v, j, iq); 
								NV_MAT_V(v, j, iq) = h + s * (g - h * tau); 
								NV_MAT_V(v, j, ip) = g - s * (h + g * tau); 
							}
						}
					}
				} else {
					/* 飛ばす */
				}
			}
		}
		for (ip = 0; ip < n; ++ip) {
			NV_MAT_V(b, ip, 0) += NV_MAT_V(z, ip, 0);
			NV_MAT_V(d, ip, 0) = NV_MAT_V(b, ip, 0);
			NV_MAT_V(z, ip, 0) = 0.0f;
		}
	}

	//if (converge) 
	{
		/* 固有値で降順ソート. 固有ベクトルを転置. */
		int j;
		nv_matrix_t *eigen = nv_matrix_alloc(eigen_vec->n + 1, eigen_vec->m);
		for (j = 0; j < eigen_vec->m; ++j) {
			/* 0:固有値 */
			NV_MAT_V(eigen, j, 0) = NV_MAT_V(d, j, 0); 
			/* 1-:固有ベクトル */
			for (i = 0; i < eigen_vec->n; ++i) {
				NV_MAT_V(eigen, j, 1 + i) = NV_MAT_V(v, i, j);
			}
		}
		qsort(eigen->v, eigen->m, NV_VEC_SIZE(eigen), sort_eigen);

		for (j = 0; j < eigen_vec->m; ++j) {
			NV_MAT_V(eigen_val, j, 0) = NV_MAT_V(eigen, j, 0);
			for (i = 0; i < eigen_vec->n; ++i) {
				NV_MAT_V(eigen_vec, j, i) = NV_MAT_V(eigen, j, 1 + i);
			}
		}
		nv_matrix_free(&eigen);
	}
	nv_matrix_free(&b);
	nv_matrix_free(&z);
	nv_matrix_free(&a);

	return converge ? 0:1;
}
