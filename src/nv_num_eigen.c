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
#include "nv_num_eigen.h"

static int 
sort_eigen(const void *p1, const void *p2)
{
	float *f1 = (float *)p1;
	float *f2 = (float *)p2;

	if (f1[0] > f2[0]) {
		return -1;
	} else if (f1[0] < f2[0]) {
		return 1;
	}

	return 0;
}


/* 巡回Jacobi法で実対称行列の固有値, 固有ベクトルを求める.
 * 参考: Numerical Recipes in C
 */
int 
nv_eigen_sym(nv_matrix_t *eigen_vec, 
			nv_matrix_t *eigen_val,
			const nv_matrix_t *smat,
			int max_epoch)
{
	int i, nrot, converge;
	const int n = smat->n;
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

	nrot = 0;
	converge = 0;
	for (i = 0; i < max_epoch; ++i) {
		/* 収束判定 */
		float sm = 0.0f;
		float tresh;
		int ip;
		
		for (ip = 0; ip < n - 1; ++ip) {
			int iq;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sm)
#endif
			for (iq = ip + 1; iq < n; ++iq) {
				sm += fabsf(NV_MAT_V(a, ip, iq));
			}
		}
		if (sm <= FLT_EPSILON) {
			converge = 1;
			break;
		}
		tresh = (i > 3)  ? 0.0f: (0.2f * sm / (n * n));

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
#pragma omp parallel sections num_threads(2)
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
					++nrot;
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
		nv_matrix_t *eigen = nv_matrix_alloc(eigen_vec->n + 1, eigen_vec->m);
		for (i = 0; i < eigen_vec->m; ++i) {
			int j;
			/* 0:固有値 */
			NV_MAT_V(eigen, i, 0) = NV_MAT_V(d, i, 0); 
			/* 1-:固有ベクトル */
			for (j = 0; j < eigen_vec->n; ++j) {
				NV_MAT_V(eigen, i, 1 + j) = NV_MAT_V(v, j, i);
			}
		}
		qsort(eigen->v, eigen->m, NV_VEC_SIZE(eigen), sort_eigen);

		for (i = 0; i < eigen_vec->m; ++i) {
			int j;
			NV_MAT_V(eigen_val, i, 0) = NV_MAT_V(eigen, i, 0);
			for (j = 0; j < eigen_vec->n; ++j) {
				NV_MAT_V(eigen_vec, i, j) = NV_MAT_V(eigen, i, 1 + j);
			}
		}
		nv_matrix_free(&eigen);
	}
	nv_matrix_free(&b);
	nv_matrix_free(&z);
	nv_matrix_free(&a);

	return converge ? 0:1;
}
