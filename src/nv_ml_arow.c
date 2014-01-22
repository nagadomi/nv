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
#include "nv_num.h"
#include "nv_ml_arow.h"

/* Multi-Class Adaptive Regularization of Weight Vectors */

static int nv_arow_progress_flag = 0;
void nv_arow_progress(int onoff)
{
	nv_arow_progress_flag = onoff;
}

nv_arow_t *
nv_arow_alloc(int n, int k)
{
	nv_arow_t *arow = (nv_arow_t *)nv_alloc_type(nv_arow_t, 1);
	NV_ASSERT(k >= 2);
	
	arow->n = n;
	arow->k = k;
	if (arow->k == 2) {
		arow->w = nv_matrix_alloc(arow->n, 1);
	} else {
		arow->w = nv_matrix_alloc(arow->n, k);
	}
	arow->bias = nv_matrix_alloc(1, k);
	
	return arow;
}

void
nv_arow_init(nv_arow_t *arow)
{
	nv_matrix_zero(arow->w);
	nv_matrix_zero(arow->bias);
}

void
nv_arow_free(nv_arow_t **arow)
{
	if (arow && *arow) {
		nv_matrix_free(&(*arow)->w);
		nv_matrix_free(&(*arow)->bias);
		nv_free(*arow);
		*arow = NULL;
	}
}

static void
nv_arow_train_at(nv_arow_t *arow,
				 const nv_matrix_t *data,
				 const nv_matrix_t *labels,
				 int posi_label,
				 float r,
				 int max_epoch)
{
#if NV_ENABLE_SSE
	const int pk_lp = (arow->n & 0xfffffffc);
	const __m128 o = _mm_set1_ps(1.0f);
	const __m128 b = _mm_set1_ps(1.0f / r);
#endif
	int epoch;
	nv_matrix_t *cov = nv_matrix_alloc(arow->n, 1);
	float covb = 1.0f;
	const float r_inv = 1.0f / r;
	
	nv_matrix_fill(cov, 1.0f);
	
	for (epoch = 0; epoch < max_epoch; ++epoch) {
		int j;
		int e = 0;
		long t = nv_clock();
		
		for (j = 0; j < data->m; ++j) {
			int i;
			int rand_j = nv_rand_index(data->m);
			int y = NV_MAT_VI(labels, rand_j, 0) == posi_label ? 1:-1;
			float xu = nv_vector_dot(arow->w, posi_label, data, rand_j) + NV_MAT_V(arow->bias, posi_label, 0);
			float alpha = 1.0f - y * xu;
			float var;
#if NV_ENABLE_SSE
			NV_ALIGNED(float, mm[4], 16);
			__m128 a, c, d, u;
#endif
			e += xu * y < 0.0f ? 1 : 0;
			
			if (alpha < 0.0f) {
				continue;
			}
			
#if NV_ENABLE_SSE
			u = _mm_setzero_ps();
			for (i = 0; i < pk_lp; i += 4) {
				d = _mm_load_ps(&NV_MAT_V(data, rand_j, i));
				c = _mm_load_ps(&NV_MAT_V(cov, 0, i));
				u = _mm_add_ps(u, _mm_mul_ps(_mm_mul_ps(d, d), c));
			}
			_mm_store_ps(mm, u);
			var = covb + mm[0] + mm[1] + mm[2] + mm[3];
			for (i = pk_lp; i < arow->n; ++i) {
				var += NV_MAT_V(data, rand_j, i) * NV_MAT_V(data, rand_j, i)
					* NV_MAT_V(cov, 0, i);
			}
			
			alpha *= (1.0f / (var + r)) * y;
			a = _mm_set1_ps(alpha);
			for (i = 0; i < pk_lp; i += 4) {
				d = _mm_load_ps(&NV_MAT_V(data, rand_j, i));
				c = _mm_load_ps(&NV_MAT_V(cov, 0, i));
				
				u = _mm_mul_ps(d, c);
				u = _mm_mul_ps(u, a);
				u = _mm_add_ps(u, _mm_load_ps(&NV_MAT_V(arow->w, posi_label, i)));
				_mm_store_ps(&NV_MAT_V(arow->w, posi_label, i), u);
				
				d = _mm_mul_ps(d, d);
				d = _mm_mul_ps(d, b);
				c = _mm_div_ps(o, c);
				c = _mm_add_ps(c, d);
				c = _mm_div_ps(o, c);
				_mm_store_ps(&NV_MAT_V(cov, 0, i), c);
			}
			for (i = pk_lp; i < arow->n; ++i) {
				NV_MAT_V(arow->w, posi_label, i) += alpha *
					NV_MAT_V(cov, 0, i) * NV_MAT_V(data, rand_j, i);
				NV_MAT_V(cov, 0, i) =
					1.0f / (1.0f / NV_MAT_V(cov, 0, i) +
							(NV_MAT_V(data, rand_j, i) 
							 * NV_MAT_V(data, rand_j, i) * r_inv));
			}
			NV_MAT_V(arow->bias, posi_label, 0) += alpha * covb;
			covb = 1.0f / ((1.0f / covb) + r_inv);
#else
			var = covb;
			for (i = 0; i < arow->n; ++i) {
				var += NV_MAT_V(data, rand_j, i) * NV_MAT_V(data, rand_j, i) *
					NV_MAT_V(cov, 0, i);
			}
			alpha *= (1.0f / (var + r)) * y;
			for (i = 0; i < arow->n; ++i) {
				NV_MAT_V(arow->w, posi_label, i) += alpha *
					NV_MAT_V(cov, 0, i) * NV_MAT_V(data, rand_j, i);
				NV_MAT_V(cov, 0, i) =
					1.0f / (1.0f / NV_MAT_V(cov, 0, i) +
							(NV_MAT_V(data, rand_j, i) 
							 * NV_MAT_V(data, rand_j, i) * (1.0f / r)));
			}
			NV_MAT_V(arow->bias, posi_label, 0) += alpha * covb;
			covb = 1.0f / ((1.0f / covb) + r_inv);
#endif
		}
		if (nv_arow_progress_flag) {
			printf("nv_arow: label: %d, epoch: %d, ER: %f, %ldms\n",
				   posi_label, epoch, (float)e / data->m, nv_clock() - t);
		}
	}
	nv_matrix_free(&cov);
}

void
nv_arow_train(nv_arow_t *arow,
			  const nv_matrix_t *data,
			  const nv_matrix_t *labels,
			  float r,
			  int max_epoch)
{
	if (arow->k == 2) {
		nv_arow_train_at(arow, data, labels, 0, r, max_epoch);
	} else {
		int k;
#ifdef _OPENMP
#pragma omp	parallel for schedule(dynamic, 1)
#endif
		for (k = 0; k < arow->k; ++k) {
			nv_arow_train_at(arow, data, labels, k, r, max_epoch);
		}
	}
}

int
nv_arow_predict_label(const nv_arow_t *arow,
					  const nv_matrix_t *vec, int j)
{
	if (arow->n == 2) {
		float y = nv_vector_dot(arow->w, 0, vec, j) + NV_MAT_V(arow->bias, 0, 0);
		return y >= 0.0f ? 0 : 1;
	} else {
		int max_k = -1;
		float max_dot = -FLT_MAX;
		int k;
		for (k = 0; k < arow->k; ++k) {
			float dot = nv_vector_dot(arow->w, k, vec, j) + NV_MAT_V(arow->bias, 0, 0);
			if (dot > max_dot) {
				max_dot = dot;
				max_k = k;
			}
		}
		return max_k;
	}
}

void
nv_arow_dump_c(FILE *out,
			   const nv_arow_t *arow,
			   const char *name, int static_variable)
{
}
