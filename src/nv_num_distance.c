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
#include "nv_num.h"
#include "nv_num_distance.h"

float nv_histgram_intersection(const nv_matrix_t *vec1, int m1, const nv_matrix_t *query_vec, int m2)
{
	float sum = nv_vector_sum(query_vec, m2);
	float section = 0;
	int n;

	if (sum == 0.0f) {
		return 1.0f;
	}
	for (n = 0; n < vec1->n; ++n) {
		section += NV_MIN(NV_MAT_V(vec1, m1, n), NV_MAT_V(query_vec, m2, n));
	}

	return 1.0f - (section / sum);
}

float nv_euclidean(const nv_matrix_t *vec1, int m1, const nv_matrix_t *vec2, int m2)
{
	return sqrtf(nv_euclidean2(vec1, m1, vec2, m2));
}

float nv_cosine(const nv_matrix_t *vec1, int m1, const nv_matrix_t *vec2, int m2)
{
	float dot = nv_vector_dot(vec1, m1, vec2, m2);
	float norm = nv_vector_norm(vec1, m1) * nv_vector_norm(vec2, m2);

	if (norm == 0.0f) {
		return 2.0f;
	}
	return 1.0f - (dot / norm);
}

/* ユークリッド距離^2 */
float nv_euclidean2(const nv_matrix_t *vec1, int m1, const nv_matrix_t *vec2, int m2)
{
	NV_ALIGNED(float, dist, 32);
	
	NV_ASSERT(vec1->n == vec2->n);
#if NV_ENABLE_AVX
	{
		__m256 x, u, h;
		__m128 a;
		int n;
		int pk_lp = (vec1->n & 0xfffffff8);
		
		u = _mm256_setzero_ps();
		for (n = 0; n < pk_lp; n += 8) {
			x = _mm256_load_ps(&NV_MAT_V(vec2, m2, n));
			h = _mm256_sub_ps(x, *(const __m256*)&NV_MAT_V(vec1, m1, n));
			x = _mm256_mul_ps(h, h);
			h = _mm256_hadd_ps(x, x);
			u = _mm256_add_ps(u, h);
		}
		u = _mm256_hadd_ps(u, u);
		a = _mm_add_ps(_mm256_extractf128_ps(u, 0), _mm256_extractf128_ps(u, 1));
		_mm_store_ss(&dist, a);
		for (n = pk_lp; n < vec1->n; ++n) {
			const float d = NV_MAT_V(vec1, m1, n) - NV_MAT_V(vec2, m2, n);			
			dist += d * d;
		}
	}
#elif NV_ENABLE_SSE2
	{
		__m128 x, u;
		int n;
		int pk_lp = (vec1->n & 0xfffffffc);
		NV_ALIGNED(float, mm[4], 16);
		u = _mm_setzero_ps();
		for (n = 0; n < pk_lp; n += 4) {
			x = _mm_sub_ps(_mm_load_ps(&NV_MAT_V(vec2, m2, n)),
						   *(const __m128 *)&NV_MAT_V(vec1, m1, n));
			u = _mm_add_ps(u, _mm_mul_ps(x, x));
		}
		_mm_store_ps(mm, u);
		dist = mm[0] + mm[1] + mm[2] + mm[3];

		for (n = pk_lp; n < vec1->n; ++n) {
			const float d = NV_MAT_V(vec1, m1, n) - NV_MAT_V(vec2, m2, n);
			dist += d * d;
		}
	}
#else
	{
		int n;
		dist = 0.0f;
		for (n = 0; n < vec1->n; ++n) {
			const float d = NV_MAT_V(vec1, m1, n) - NV_MAT_V(vec2, m2, n);			
			dist += d * d;
		}
	}
#endif
	return dist;
}

