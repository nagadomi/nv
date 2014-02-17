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
#include "nv_num_distance.h"
#include "nv_num_vector.h"

void 
nv_vector_sqrt(nv_matrix_t *vec0, int m0,
			   const nv_matrix_t *vec1, int m1)
{
	int i;
	
	NV_ASSERT(vec0->n == vec1->n);
	
	for (i = 0; i < vec0->n; ++i) {
		NV_MAT_V(vec0, m0, i) = sqrtf(NV_MAT_V(vec1, m1, i));
	}
}

void 
nv_vector_not10(nv_matrix_t *vec0, int m0,
				const nv_matrix_t *vec1, int m1)
{
	int i;

	NV_ASSERT(vec0->n == vec1->n);
	for (i = 0; i < vec0->n; ++i) {
		if (NV_MAT_V(vec1, m1, i) != 0.0f) {
			NV_MAT_V(vec0, m0, i) = 0.0f;
		} else {
			NV_MAT_V(vec0, m0, i) = 1.0f;
		}
	}
}

void 
nv_vector_pows(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  float x)
{
	int n;

	NV_ASSERT(vec1->n == vec1->n);
	
	for (n = 0; n < vec0->n; ++n) {
		NV_MAT_V(vec0, m0, n) = powf(NV_MAT_V(vec1, m1, n), x);
	}
}


void
nv_vector_in_range10(nv_matrix_t *dst,
					 int dj,
					 const nv_matrix_t *lower,
					 int lj,
					 const nv_matrix_t *upper,
					 int uj,
					 const nv_matrix_t *vec,
					 int vj)
{
	int i;

	NV_ASSERT(dst->n == lower->n && upper->n == dst->n);

	for (i = 0; i < vec->n; ++i) {
		if (NV_MAT_V(lower, lj, i) <= NV_MAT_V(vec, vj, i) &&
			NV_MAT_V(vec, vj, i) <= NV_MAT_V(upper, uj, i))
		{
			NV_MAT_V(dst, dj, i) = 1.0f;
		} else {
			NV_MAT_V(dst, dj, i) = 0.0f;
		}
	}
}

void 
nv_vector_subsm(nv_matrix_t *vec0, int m0,
				const nv_matrix_t *vec1, int m1,
				float v,
				const nv_matrix_t *mask, int m2
	)
{
	int i;

	NV_ASSERT(vec1->n == vec0->n);
	
	for (i = 0; i < vec1->n; ++i) {
		if (NV_MAT_V(mask, m2, i) != 0.0f) {
			NV_MAT_V(vec0, m0, i) = NV_MAT_V(vec1, m1, i) - v;
		}
	}
}

void 
nv_vector_subs(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  float v)
{
	NV_ASSERT(vec1->n == vec0->n);
	
#if NV_ENABLE_SSE2
	{
		__m128 vv;
		int n;
		int pk_lp = (vec1->n & 0xfffffffc);
		vv = _mm_set1_ps(v);
		for (n = 0; n < pk_lp; n += 4) {
			__m128 x = _mm_load_ps(&NV_MAT_V(vec1, m1, n));
			x = _mm_sub_ps(x, vv);
			_mm_store_ps(&NV_MAT_V(vec0, m0, n), x);
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) - v;
		}
	}
#else
	{
		int n;
		for (n = 0; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) - v;
		}
	}
#endif
}

void 
nv_vector_sub(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  const nv_matrix_t *vec2, int m2)
{
	NV_ASSERT(vec1->n == vec2->n);
	NV_ASSERT(vec2->n == vec0->n);
	
#if NV_ENABLE_AVX	
	{
		__m256 x;
		int n;
		int pk_lp = (vec1->n & 0xfffffff8);

		for (n = 0; n < pk_lp; n += 8) {
			x = _mm256_load_ps(&NV_MAT_V(vec1, m1, n));
			x = _mm256_sub_ps(x, *(const __m256 *)&NV_MAT_V(vec2, m2, n));
			_mm256_store_ps(&NV_MAT_V(vec0, m0, n), x);
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) - NV_MAT_V(vec2, m2, n);
		}
	}
#elif NV_ENABLE_SSE2
	{
		int n;
		int pk_lp = (vec1->n & 0xfffffffc);

		for (n = 0; n < pk_lp; n += 4) {
			__m128 x = _mm_load_ps(&NV_MAT_V(vec1, m1, n));
			x = _mm_sub_ps(x, *(const __m128 *)&NV_MAT_V(vec2, m2, n));
			_mm_store_ps(&NV_MAT_V(vec0, m0, n), x);
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) - NV_MAT_V(vec2, m2, n);
		}
	}
#else
	{
		int n;
		for (n = 0; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) - NV_MAT_V(vec2, m2, n);
		}
	}
#endif
}

void 
nv_vector_adds(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			   float v)
{
	NV_ASSERT(vec1->n == vec0->n);
	
#if NV_ENABLE_SSE2
	{
		__m128 vv;
		int n;
		int pk_lp = (vec1->n & 0xfffffffc);
		
		vv = _mm_set1_ps(v);

		for (n = 0; n < pk_lp; n += 4) {
			__m128 x = _mm_load_ps(&NV_MAT_V(vec1, m1, n));
			_mm_store_ps(&NV_MAT_V(vec0, m0, n),
						 _mm_add_ps(x, vv));
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) + v;
		}
	}
#else
	{
		int n;
		for (n = 0; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) + v;
		}
	}
#endif
}

void 
nv_vector_add(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  const nv_matrix_t *vec2, int m2)
{
	NV_ASSERT(vec1->n == vec2->n);
	NV_ASSERT(vec2->n == vec0->n);
	
#if NV_ENABLE_AVX
	{
		__m256 x;
		int n;
		int pk_lp = (vec1->n & 0xfffffff8);
		
		for (n = 0; n < pk_lp; n += 8) {
			x = _mm256_load_ps(&NV_MAT_V(vec1, m1, n));
			_mm256_store_ps(&NV_MAT_V(vec0, m0, n),
							_mm256_add_ps(x, *(const __m256 *)&NV_MAT_V(vec2, m2, n)));
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) + NV_MAT_V(vec2, m2, n);
		}
	}
#elif NV_ENABLE_SSE2
	{
		
		int n;
		int pk_lp = (vec1->n & 0xfffffffc);

		for (n = 0; n < pk_lp; n += 4) {
			__m128 x = _mm_load_ps(&NV_MAT_V(vec1, m1, n));
			_mm_store_ps(&NV_MAT_V(vec0, m0, n),
						 _mm_add_ps(x, *(const __m128 *)&NV_MAT_V(vec2, m2, n)));
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) + NV_MAT_V(vec2, m2, n);
		}
	}
#else
	{
		int n;
		for (n = 0; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) + NV_MAT_V(vec2, m2, n);
		}
	}
#endif
}

void 
nv_vector_max(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  const nv_matrix_t *vec2, int m2)
{
	NV_ASSERT(vec1->n == vec2->n);
	NV_ASSERT(vec2->n == vec0->n);
	
#if NV_ENABLE_AVX
	{
		__m256 x;
		int n;
		int pk_lp = (vec1->n & 0xfffffff8);
		
		for (n = 0; n < pk_lp; n += 8) {
			x = _mm256_load_ps(&NV_MAT_V(vec1, m1, n));
			_mm256_store_ps(&NV_MAT_V(vec0, m0, n),
							_mm256_max_ps(x, *(const __m256 *)&NV_MAT_V(vec2, m2, n)));
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) > NV_MAT_V(vec2, m2, n) ?
				NV_MAT_V(vec1, m1, n):NV_MAT_V(vec2, m2, n);
		}
	}
#elif NV_ENABLE_SSE2
	{
		
		int n;
		int pk_lp = (vec1->n & 0xfffffffc);

		for (n = 0; n < pk_lp; n += 4) {
			__m128 x = _mm_load_ps(&NV_MAT_V(vec1, m1, n));
			_mm_store_ps(&NV_MAT_V(vec0, m0, n),
						 _mm_max_ps(x, *(const __m128 *)&NV_MAT_V(vec2, m2, n)));
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) > NV_MAT_V(vec2, m2, n) ?
				NV_MAT_V(vec1, m1, n):NV_MAT_V(vec2, m2, n);
		}
	}
#else
	{
		int n;
		for (n = 0; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) > NV_MAT_V(vec2, m2, n) ?
				NV_MAT_V(vec1, m1, n):NV_MAT_V(vec2, m2, n);
		}
	}
#endif
}
void 
nv_vector_min(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  const nv_matrix_t *vec2, int m2)
{
	NV_ASSERT(vec1->n == vec2->n);
	NV_ASSERT(vec2->n == vec0->n);
	
#if NV_ENABLE_AVX
	{
		__m256 x;
		int n;
		int pk_lp = (vec1->n & 0xfffffff8);
		
		for (n = 0; n < pk_lp; n += 8) {
			x = _mm256_load_ps(&NV_MAT_V(vec1, m1, n));
			_mm256_store_ps(&NV_MAT_V(vec0, m0, n),
							_mm256_min_ps(x, *(const __m256 *)&NV_MAT_V(vec2, m2, n)));
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) < NV_MAT_V(vec2, m2, n) ?
				NV_MAT_V(vec1, m1, n):NV_MAT_V(vec2, m2, n);
		}
	}
#elif NV_ENABLE_SSE2
	{
		
		int n;
		int pk_lp = (vec1->n & 0xfffffffc);

		for (n = 0; n < pk_lp; n += 4) {
			__m128 x = _mm_load_ps(&NV_MAT_V(vec1, m1, n));
			_mm_store_ps(&NV_MAT_V(vec0, m0, n),
						 _mm_min_ps(x, *(const __m128 *)&NV_MAT_V(vec2, m2, n)));
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) < NV_MAT_V(vec2, m2, n) ?
				NV_MAT_V(vec1, m1, n):NV_MAT_V(vec2, m2, n);
		}
	}
#else
	{
		int n;
		for (n = 0; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) < NV_MAT_V(vec2, m2, n) ?
				NV_MAT_V(vec1, m1, n):NV_MAT_V(vec2, m2, n);
		}
	}
#endif
}


void 
nv_vector_mul(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  const nv_matrix_t *vec2, int m2)
{
	NV_ASSERT(vec1->n == vec2->n);
	NV_ASSERT(vec2->n == vec0->n);
	
#if NV_ENABLE_AVX
	{
		__m256 x;
		int n;
		int pk_lp = (vec1->n & 0xfffffff8);
		
		for (n = 0; n < pk_lp; n += 8) {
			x = _mm256_load_ps(&NV_MAT_V(vec1, m1, n));
			_mm256_store_ps(&NV_MAT_V(vec0, m0, n),
							_mm256_mul_ps(x, *(const __m256 *)&NV_MAT_V(vec2, m2, n)));
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) * NV_MAT_V(vec2, m2, n);
		}
	}
#elif NV_ENABLE_SSE2
	{
		int n;
		int pk_lp = (vec1->n & 0xfffffffc);

		for (n = 0; n < pk_lp; n += 4) {
			__m128 x = _mm_load_ps(&NV_MAT_V(vec1, m1, n));
			_mm_store_ps(&NV_MAT_V(vec0, m0, n),
						 _mm_mul_ps(x, *(const __m128 *)&NV_MAT_V(vec2, m2, n)));
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) * NV_MAT_V(vec2, m2, n);
		}
	}
#else
	{
		int n;
		for (n = 0; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) * NV_MAT_V(vec2, m2, n);
		}
	}
#endif
}

void 
nv_vector_div(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  const nv_matrix_t *vec2, int m2)
{
	NV_ASSERT(vec1->n == vec2->n);
	NV_ASSERT(vec2->n == vec0->n);
	
#if NV_ENABLE_AVX
	{
		__m256 x;
		int n;
		int pk_lp = (vec1->n & 0xfffffff8);
		
		for (n = 0; n < pk_lp; n += 8) {
			x = _mm256_load_ps(&NV_MAT_V(vec1, m1, n));
			_mm256_store_ps(&NV_MAT_V(vec0, m0, n),
							_mm256_div_ps(x, *(const __m256 *)&NV_MAT_V(vec2, m2, n)));
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) / NV_MAT_V(vec2, m2, n);
		}
	}
#elif NV_ENABLE_SSE2
	{
		int n;
		int pk_lp = (vec1->n & 0xfffffffc);

		for (n = 0; n < pk_lp; n += 4) {
			__m128 x = _mm_load_ps(&NV_MAT_V(vec1, m1, n));
			_mm_store_ps(&NV_MAT_V(vec0, m0, n),
						 _mm_div_ps(x, *(const __m128 *)&NV_MAT_V(vec2, m2, n)));
		}
		for (n = pk_lp; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) / NV_MAT_V(vec2, m2, n);
		}
	}
#else
	{
		int n;
		for (n = 0; n < vec1->n; ++n) {
			NV_MAT_V(vec0, m0, n) = NV_MAT_V(vec1, m1, n) / NV_MAT_V(vec2, m2, n);
		}
	}
#endif
}



float 
nv_vector_dot(const nv_matrix_t *vec1, int m1,
			  const nv_matrix_t *vec2, int m2)
{
	NV_ASSERT(vec1->n == vec2->n);
	
#if NV_ENABLE_AVX
	{
		__m256 x, u, h;
		__m128 a;
		int n;
		int pk_lp = (vec1->n & 0xfffffff8);
		NV_ALIGNED(float, dp, 32);

		u = _mm256_setzero_ps();
		for (n = 0; n < pk_lp; n += 8) {
			x = _mm256_load_ps(&NV_MAT_V(vec2, m2, n));
			h = _mm256_load_ps(&NV_MAT_V(vec1, m1, n));
			x = _mm256_mul_ps(x, h);
			h = _mm256_hadd_ps(x, x);
			u = _mm256_add_ps(u, h);
		}
		u = _mm256_hadd_ps(u, u);
		a = _mm_add_ps(_mm256_extractf128_ps(u, 0), _mm256_extractf128_ps(u, 1));
		_mm_store_ss(&dp, a);
		for (n = pk_lp; n < vec1->n; ++n) {
			dp += NV_MAT_V(vec1, m1, n) * NV_MAT_V(vec2, m2, n);
		}
		
		return dp;
	}
#elif NV_ENABLE_SSE2
	{
		__m128 x, u;
		int n;
		int pk_lp = (vec1->n & 0xfffffffc);
		NV_ALIGNED(float, dp, 16);
		NV_ALIGNED(float, mm[4], 16);
		u = _mm_setzero_ps();
		for (n = 0; n < pk_lp; n += 4) {
			x = _mm_load_ps(&NV_MAT_V(vec2, m2, n));
			u = _mm_add_ps(u,
						   _mm_mul_ps(x, *(__m128 *)&NV_MAT_V(vec1, m1, n)));
		}
		_mm_store_ps(mm, u);
		dp = mm[0] + mm[1] + mm[2] + mm[3];
		for (n = pk_lp; n < vec1->n; ++n) {
			dp += NV_MAT_V(vec1, m1, n) * NV_MAT_V(vec2, m2, n);
		}
		return dp;
	}
#else
	{
		int n;
		float dp = 0.0f;
		for (n = 0; n < vec1->n; ++n) {
			dp += NV_MAT_V(vec1, m1, n) * NV_MAT_V(vec2, m2, n);
		}
		return dp;
	}
#endif
}

float 
nv_vector_norm(const nv_matrix_t *vec, int vec_m)
{
#if NV_ENABLE_AVX
	{
		NV_ALIGNED(float, mm[8], 32);
		__m256 x, u;
		int n;
		int pk_lp = (vec->n & 0xfffffff8);
		float dp = 0.0f;
		
		u = _mm256_setzero_ps();
		for (n = 0; n < pk_lp; n += 8) {
			x = _mm256_load_ps(&NV_MAT_V(vec, vec_m, n));
			u = _mm256_add_ps(u, _mm256_mul_ps(x, x));
		}
		_mm256_store_ps(mm, u);
		dp = mm[0] + mm[1] + mm[2] + mm[3] + mm[4] + mm[5] + mm[6] + mm[7];
		for (n = pk_lp; n < vec->n; ++n) {
			dp += NV_MAT_V(vec, vec_m, n) * NV_MAT_V(vec, vec_m, n);
		}
		if (dp > 0.0f) {
			return sqrtf(dp);
		}
		return 0.0f;
	}
#elif NV_ENABLE_SSE2
	{
		NV_ALIGNED(float, mm[4], 16);
		__m128 x, u;
		int n;
		int pk_lp = (vec->n & 0xfffffffc);
		float dp = 0.0f;
		
		u = _mm_setzero_ps();
		for (n = 0; n < pk_lp; n += 4) {
			x = _mm_load_ps(&NV_MAT_V(vec, vec_m, n));
			u = _mm_add_ps(u, _mm_mul_ps(x, x));
		}
		_mm_store_ps(mm, u);
		dp = mm[0] + mm[1] + mm[2] + mm[3];
		for (n = pk_lp; n < vec->n; ++n) {
			dp += NV_MAT_V(vec, vec_m, n) * NV_MAT_V(vec, vec_m, n);
		}
		if (dp > 0.0f) {
			return sqrtf(dp);
		}
		return 0.0f;
	}
#else
	{
		int n;
		float dp = 0.0f;
		for (n = 0; n < vec->n; ++n) {
			dp += NV_MAT_V(vec, vec_m, n) * NV_MAT_V(vec, vec_m, n);
		}
		if (dp > 0.0f) {
			return sqrtf(dp);
		}
		return 0.0f;
	}
#endif
}

nv_int_float_t
nv_vector_max_ex(const nv_matrix_t *v, int m)
{
	nv_int_float_t ret;
	ret.f = nv_vector_maxs(v, m);
	ret.i = (int)nv_float_find_index(&NV_MAT_V(v, m, 0), 0, v->n, ret.f);
	
	return ret;
}

int 
nv_vector_max_n(const nv_matrix_t *v, int m)
{
	float v_max = nv_vector_maxs(v, m);
	int max_n = (int)nv_float_find_index(&NV_MAT_V(v, m, 0), 0, v->n, v_max);
	return max_n;
}

float 
nv_vector_maxs(const nv_matrix_t *v, int j)
{
	float v_max = -FLT_MAX;
	int i;
#if NV_ENABLE_AVX
	{
		NV_ALIGNED(float, mm[9], 32);
		__m256 max_vec;
		int pk_lp = (v->n & 0xfffffff8);
		
		max_vec = _mm256_set1_ps(-FLT_MAX);
		for (i = 0; i < pk_lp; i += 8) {
			max_vec = _mm256_max_ps(max_vec, *(const __m256 *)&NV_MAT_V(v, j, i));
		}
		_mm256_store_ps(mm, max_vec);
		
		for (i = pk_lp; i < v->n; ++i) {
			if (NV_MAT_V(v, j, i) > v_max) {
				v_max = NV_MAT_V(v, j, i);
			}
		}
		mm[8] = v_max;
		for (i = 0; i < 9; ++i) {
			if (mm[i] > v_max) {
				v_max = mm[i];
			}
		}
	}
#elif NV_ENABLE_SSE2
	{
		NV_ALIGNED(float, mm[5], 16);
		__m128 max_vec;
		int pk_lp = (v->n & 0xfffffffc);
		
		max_vec = _mm_set1_ps(-FLT_MAX);
		for (i = 0; i < pk_lp; i += 4) {
			max_vec = _mm_max_ps(max_vec, *(const __m128 *)&NV_MAT_V(v, j, i));
		}
		_mm_store_ps(mm, max_vec);
		
		for (i = pk_lp; i < v->n; ++i) {
			if (NV_MAT_V(v, j, i) > v_max) {
				v_max = NV_MAT_V(v, j, i);
			}
		}
		mm[4] = v_max;
		for (i = 0; i < 5; ++i) {
			if (mm[i] > v_max) {
				v_max = mm[i];
			}
		}
	}
#else
	for (i = 0; i < v->n; ++i) {
		if (NV_MAT_V(v, j, i) > v_max) {
			v_max = NV_MAT_V(v, j, i);
		}
	}
#endif
	return v_max;
}

float
nv_vector_mins(const nv_matrix_t *v, int j)
{
	float v_min = FLT_MAX;
	int i;
#if NV_ENABLE_AVX
	{
		NV_ALIGNED(float, mm[9], 32);
		__m256 min_vec;
		int pk_lp = (v->n & 0xfffffff8);

		min_vec = _mm256_set1_ps(FLT_MAX);
		for (i = 0; i < pk_lp; i += 8) {
			min_vec = _mm256_min_ps(min_vec, *(const __m256 *)&NV_MAT_V(v, j, i));
		}
		_mm256_store_ps(mm, min_vec);
		
		for (i = pk_lp; i < v->n; ++i) {
			if (NV_MAT_V(v, j, i) < v_min) {
				v_min = NV_MAT_V(v, j, i);
			}
		}
		mm[8] = v_min;
		for (i = 0; i < 9; ++i) {
			if (mm[i] < v_min) {
				v_min = mm[i];
			}
		}
	}
#elif NV_ENABLE_SSE2
	{
		NV_ALIGNED(float, mm[5], 16);
		__m128 min_vec;
		int pk_lp = (v->n & 0xfffffffc);

		min_vec = _mm_set1_ps(FLT_MAX);
		for (i = 0; i < pk_lp; i += 4) {
			min_vec = _mm_min_ps(min_vec, *(const __m128 *)&NV_MAT_V(v, j, i));
		}
		_mm_store_ps(mm, min_vec);

		for (i = pk_lp; i < v->n; ++i) {
			if (NV_MAT_V(v, j, i) < v_min) {
				v_min = NV_MAT_V(v, j, i);
			}
		}
		mm[4] = v_min;
		for (i = 0; i < 5; ++i) {
			if (mm[i] < v_min) {
				v_min = mm[i];
			}
		}
	}
#else
	for (i = 0; i < v->n; ++i) {
		if (NV_MAT_V(v, j, i) < v_min) {
			v_min = NV_MAT_V(v, j, i);
		}
	}
#endif
	return v_min;
}

nv_int_float_t
nv_vector_min_ex(const nv_matrix_t *v, int m)
{
	nv_int_float_t ret;
	ret.f = nv_vector_mins(v, m);
	ret.i = (int)nv_float_find_index(&NV_MAT_V(v, m, 0), 0, v->n, ret.f);
	
	return ret;
}


int 
nv_vector_min_n(const nv_matrix_t *v, int m)
{
	float v_min = nv_vector_mins(v, m);
	int min_n = (int)nv_float_find_index(&NV_MAT_V(v, m, 0), 0, v->n, v_min);
	return min_n;
}

int 
nv_vector_maxnorm_m(const nv_matrix_t *v)
{
	int m, max_m = -1;
	float v_max = -FLT_MAX;

	for (m = 0; m < v->m; ++m) {
		float norm = nv_vector_norm(v, m);
		if (norm > v_max) {
			max_m = m;
			v_max = norm;
		}
	}
	return max_m;
}

int 
nv_vector_minnorm_m(const nv_matrix_t *v)
{
	int m, min_m = -1;
	float v_min = FLT_MAX;

	for (m = 0; m < v->m; ++m) {
		float norm = nv_vector_norm(v, m);
		if (norm < v_min) {
			min_m = m;
			v_min = norm;
		}
	}
	return min_m;
}

float 
nv_vector_sum(const nv_matrix_t *v, int m)
{
	int n;
	float sum = 0.0f;
	for (n = 0; n < v->n; ++n) {
		sum += NV_MAT_V(v, m, n);
	}
	return sum;
}

float
nv_vector_mean(const nv_matrix_t *x, int j)
{
	return nv_vector_sum(x, j) / x->n;
}

float
nv_vector_var(const nv_matrix_t *x, int j)
{
	return nv_vector_var_ex(x, j, nv_vector_mean(x, j));
}

float
nv_vector_var_ex(const nv_matrix_t *x, int j, float mean)
{
	float var = 0.0f;
	int i;
	for (i = 0; i < x->n; ++i) {
		var += (mean - NV_MAT_V(x, j, i)) * (mean - NV_MAT_V(x, j, i));
	}
	return x->n > 1 ? (var / (x->n - 1)) : var;
}

int 
nv_vector_maxsum_m(const nv_matrix_t *v)
{
	int m, max_m = -1;
	float v_max = -FLT_MAX;

	for (m = 0; m < v->m; ++m) {
		float sum = nv_vector_sum(v, m);
		if (sum > v_max) {
			max_m = m;
			v_max = sum;
		}
	}
	return max_m;
}

int 
nv_vector_minsum_m(const nv_matrix_t *v)
{
	int m, min_m = -1;
	float v_min = FLT_MAX;

	for (m = 0; m < v->m; ++m) {
		float sum = nv_vector_sum(v, m);
		if (sum < v_min) {
			min_m = m;
			v_min = sum;
		}
	}
	return min_m;
}

void 
nv_vector_avg(nv_matrix_t *mean, int mean_m, const nv_matrix_t *mat)
{
	float factor = 1.0f / mat->m;
	int m;

	NV_ASSERT(mean->n == mat->n);

	nv_vector_zero(mean, mean_m);
	for (m = 0; m < mat->m; ++m) {
		int n;
		for (n = 0; n < mat->n; ++n) {
			NV_MAT_V(mean, mean_m, n) += factor * NV_MAT_V(mat, m, n);
		}
	}
}

void 
nv_vector_normalize_L1(nv_matrix_t *v, int vm)
{
	int i;
	float sum = 0.0f;
	for (i = 0; i < v->n; ++i) {
		sum += fabsf(NV_MAT_V(v, vm, i));
	}
	if (sum != 0.0f) {
		float scale = 1.0f / sum;
		nv_vector_muls(v, vm, v, vm, scale);
	}
}

void 
nv_vector_normalize_L2(nv_matrix_t *v, int vm)
{
#if NV_ENABLE_AVX
	{
		const int i_lp = (v->n & 0xfffffff8);
		__m256 x, u, h;
		__m128 a;
		int i;
		NV_ALIGNED(float, dp, 32);
		u = _mm256_setzero_ps();
		for (i = 0; i < i_lp; i += 8) {
			x = _mm256_load_ps(&NV_MAT_V(v, vm, i));
			x = _mm256_mul_ps(x, x);
			h = _mm256_hadd_ps(x, x);
			u = _mm256_add_ps(u, h);
		}
		u = _mm256_hadd_ps(u, u);
		a = _mm_add_ps(_mm256_extractf128_ps(u, 0), _mm256_extractf128_ps(u, 1));
		_mm_store_ss(&dp, a);
		for (i = i_lp; i < v->n; ++i) {
			dp += NV_MAT_V(v, vm, i) * NV_MAT_V(v, vm, i);
		}
		if (dp > 0.0f) {
			float scale = 1.0f / sqrtf(dp);
			x = _mm256_set1_ps(scale);
			for (i = 0; i < i_lp; i += 8) {
				_mm256_store_ps(&NV_MAT_V(v, vm, i),
								_mm256_mul_ps(*(const __m256*)&NV_MAT_V(v, vm, i), x));
			}
			for (i = i_lp; i < v->n; ++i) {
				NV_MAT_V(v, vm, i) *= scale;
			}
		}
	}
#elif NV_ENABLE_SSE
	{
		const int i_lp = (v->n & 0xfffffffc);
		__m128 x, u;
		int i;
		NV_ALIGNED(float, dp, 16);
#if NV_ENABLE_SSE3
		u = _mm_setzero_ps();
		for (i = 0; i < i_lp; i += 4) {
			x = _mm_load_ps(&NV_MAT_V(v, vm, i));
			x = _mm_mul_ps(x, x);
			x = _mm_hadd_ps(x, x);
			u = _mm_add_ps(u, x);
		}
		u = _mm_hadd_ps(u, u);
		_mm_store_ss(&dp, u);
#else
		NV_ALIGNED(float, mm[4], 16);
		u = _mm_setzero_ps();
		for (i = 0; i < i_lp; i += 4) {
			x = _mm_load_ps(&NV_MAT_V(v, vm, i));
			u = _mm_add_ps(u, _mm_mul_ps(x, x));
		}
		_mm_store_ps(mm, u);
		dp = mm[0] + mm[1] + mm[2] + mm[3];
#endif
		for (i = i_lp; i < v->n; ++i) {
			dp += NV_MAT_V(v, vm, i) * NV_MAT_V(v, vm, i);
		}
		if (dp > 0.0f) {
			float scale = 1.0f / sqrtf(dp);
			x = _mm_set1_ps(scale);
			for (i = 0; i < i_lp; i += 4) {
				_mm_store_ps(&NV_MAT_V(v, vm, i),
							 _mm_mul_ps(*(const __m128*)&NV_MAT_V(v, vm, i), x));
			}
			for (i = i_lp; i < v->n; ++i) {
				NV_MAT_V(v, vm, i) *= scale;
			}
		}
	}
#else
	float norm = nv_vector_norm(v, vm);
	if (norm > 0.0f) {
		float scale = 1.0f / norm;
		nv_vector_muls(v, vm, v, vm, scale);
	}
#endif
}

void 
nv_vector_normalize_all_L1(nv_matrix_t *mat)
{
	int j;
	
	for (j = 0; j < mat->m; ++j) {
		int i;
		float sum = 0.0f;
		
		for (i = 0; i < mat->n; ++i) {
			sum += fabsf(NV_MAT_V(mat, j, i));
		}
		if (sum != 0.0f) {
			float scale = 1.0f / sum;
			nv_vector_muls(mat, j, mat, j, scale);
		}
	}
}

void 
nv_vector_normalize_all_L2(nv_matrix_t *mat)
{
	int j;
	
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (j = 0; j < mat->m; ++j) {
		nv_vector_normalize_L2(mat, j);
	}
}

void 
nv_vector_normalize_shift(nv_matrix_t *v, int vm, float min_v, float max_v)
{
	int n;
	float cur_max_v = -FLT_MAX;
	float cur_min_v = FLT_MAX;

	for (n = 0; n < v->n; ++n) {
		if (NV_MAT_V(v, vm, n) > cur_max_v) {
			cur_max_v = NV_MAT_V(v, vm, n);
		}
		if (NV_MAT_V(v, vm, n) < cur_min_v) {
			cur_min_v = NV_MAT_V(v, vm, n);
		}
	}

	if (fabsf(cur_max_v - cur_min_v) > FLT_EPSILON) {
		float scale = (max_v - min_v) / (cur_max_v - cur_min_v);
		for (n = 0; n < v->n; ++n) {
			NV_MAT_V(v, vm, n) = (NV_MAT_V(v, vm, n) - cur_min_v) * scale + min_v;
		}
	}
}

void 
nv_vector_muls(nv_matrix_t *a, int am, const nv_matrix_t *x, int xm, float v)
{
	NV_ASSERT(a->n >= x->n);
#if NV_ENABLE_AVX
	{
		__m256 vv;
		int n;
		int pk_lp = (x->n & 0xfffffff8);

		vv = _mm256_set1_ps(v);
		for (n = 0; n < pk_lp; n += 8) {
			_mm256_store_ps(&NV_MAT_V(a, am, n),
				_mm256_mul_ps(vv, *(const __m256 *)&NV_MAT_V(x, xm, n)));
		}
		for (n = pk_lp; n < x->n; ++n) {
			NV_MAT_V(a, am, n) = NV_MAT_V(x, xm, n) * v;
		}
	}
#elif NV_ENABLE_SSE2
	{
		__m128 vv;
		int n;
		int pk_lp = (x->n & 0xfffffffc);

		vv = _mm_set1_ps(v);
		for (n = 0; n < pk_lp; n += 4) {
			_mm_store_ps(&NV_MAT_V(a, am, n),
				_mm_mul_ps(vv, *(const __m128 *)&NV_MAT_V(x, xm, n)));
		}
		for (n = pk_lp; n < x->n; ++n) {
			NV_MAT_V(a, am, n) = NV_MAT_V(x, xm, n) * v;
		}
	}
#else
	{
		int n;
		for (n = 0; n < x->n; ++n) {
			NV_MAT_V(a, am, n) = NV_MAT_V(x, xm, n) * v;
		}
	}
#endif
}

void 
nv_vector_divs(nv_matrix_t *a, int am, const nv_matrix_t *x, int xm, float v)
{
	NV_ASSERT(a->n >= x->n);
#if NV_ENABLE_AVX
	{
		__m256 vv;
		int n;
		int pk_lp = (x->n & 0xfffffff8);

		vv = _mm256_set1_ps(v);
		for (n = 0; n < pk_lp; n += 8) {
			_mm256_store_ps(&NV_MAT_V(a, am, n),
							_mm256_div_ps(*(const __m256 *)&NV_MAT_V(x, xm, n), vv));
		}
		for (n = pk_lp; n < x->n; ++n) {
			NV_MAT_V(a, am, n) = NV_MAT_V(x, xm, n) / v;
		}
	}
#elif NV_ENABLE_SSE2
	{
		__m128 vv;
		int n;
		int pk_lp = (x->n & 0xfffffffc);

		vv = _mm_set1_ps(v);
		for (n = 0; n < pk_lp; n += 4) {
			_mm_store_ps(&NV_MAT_V(a, am, n),
						 _mm_div_ps(*(const __m128 *)&NV_MAT_V(x, xm, n), vv));
		}
		for (n = pk_lp; n < x->n; ++n) {
			NV_MAT_V(a, am, n) = NV_MAT_V(x, xm, n) / v;
		}
	}
#else
	{
		int n;
		for (n = 0; n < x->n; ++n) {
			NV_MAT_V(a, am, n) = NV_MAT_V(x, xm, n) / v;
		}
	}
#endif
}

void 
nv_vector_inv(nv_matrix_t *a, int am, const nv_matrix_t *x, int xm)
{
	NV_ASSERT(a->n >= x->n);
#if NV_ENABLE_AVX
	{
		__m256 xx, vv;
		int n;
		int pk_lp = (x->n & 0xfffffff8);

		vv = _mm256_set1_ps(1.0f);

		for (n = 0; n < pk_lp; n += 8) {
			xx = _mm256_load_ps(&NV_MAT_V(x, xm, n));
			xx = _mm256_div_ps(vv, xx);
			_mm256_store_ps(&NV_MAT_V(a, am, n), xx);
		}
		for (n = pk_lp; n < x->n; ++n) {
			NV_MAT_V(a, am, n) = 1.0f / NV_MAT_V(x, xm, n);
		}
	}
#elif NV_ENABLE_SSE2
	{
		__m128 xx, vv;
		int n;
		int pk_lp = (x->n & 0xfffffffc);

		vv = _mm_set1_ps(1.0f);

		for (n = 0; n < pk_lp; n += 4) {
			xx = _mm_load_ps(&NV_MAT_V(x, xm, n));
			xx = _mm_div_ps(vv, xx);
			_mm_store_ps(&NV_MAT_V(a, am, n), xx);
		}
		for (n = pk_lp; n < x->n; ++n) {
			NV_MAT_V(a, am, n) = 1.0f / NV_MAT_V(x, xm, n);
		}
	}
#else
	{
		int n;
		for (n = 0; n < x->n; ++n) {
			NV_MAT_V(a, am, n) = 1.0f / NV_MAT_V(x, xm, n);
		}
	}
#endif
}

int64_t
nv_float_nonzero_index(const float *v, int64_t s, int64_t e)
{
#if NV_ENABLE_SSE2
	{
		__m128 zero, xmm;
		int64_t i = s, e2;
		int eq;
		
		if (s & 0x3) {
			e2 = (s & 0xfffffffffffffffcLL) + 4;
			for (; i < e2 && i < e; ++i) {
				if (v[i] != 0.0f) {
					return i;
				}
			}
		}
		zero = _mm_setzero_ps();
		e2 = (e & 0xfffffffffffffffcLL);
		for (; i < e2; i += 4) {
			xmm = _mm_load_ps(&v[i]);
			xmm = _mm_cmpneq_ps(zero, xmm);
			eq = _mm_movemask_ps(xmm);
			if (eq != 0) {
				if (eq & 0x03) {
					if (eq & 0x1) {
						return i;
					} else {
						return i + 1;
					}
				} else {
					if (eq & 0x4) {
						return i + 2;
					} else {
						return i + 3;
					}
				}
			}
		}
		for (;i < e; ++i) {
			if (v[i] != 0.0f) {
				return i;
			}
		}
		return -1;
	}
#else
	{
		int64_t i;
		for (i = s; i < e; ++i) {
			if (v[i] != 0.0f) {
				return i;
			}
		}
		return -1;
	}
#endif
}

int
nv_vector_eq(const nv_matrix_t *vec1, int j1, const nv_matrix_t *vec2, int j2)
{
	NV_ASSERT(vec1->n == vec2->n);
	
#if NV_ENABLE_SSE2
	{
		__m128 xmm;
		int i = 0;
		int eq;
		int pk_lp = (vec1->n & 0xfffffffc);
		
		for (i = 0; i < pk_lp; i += 4) {
			xmm = _mm_load_ps(&NV_MAT_V(vec2, j2, i));
			xmm = _mm_cmpneq_ps(xmm, *(const __m128 *)&NV_MAT_V(vec1, j1, i));
			eq = _mm_movemask_ps(xmm);
			if (eq != 0) {
				return 0;
			}
		}
		for (i = pk_lp; i < vec1->n; ++i) {
			if (NV_MAT_V(vec1, j1, i) != NV_MAT_V(vec2, j2, i)) {			
				return 0;
			}
		}
		return 1;
	}
#else
	{
		int i;
		for (i = 0; i < vec1->n; ++i) {
			if (NV_MAT_V(vec1, j1, i) != NV_MAT_V(vec2, j2, i)) {
				return 0;
			}
		}
		return 1;
	}
#endif
}

int64_t
nv_float_find_index(const float *v, int64_t s, int64_t e, float key)
{
#if NV_ENABLE_SSE2
	{
		__m128 xkey, xmm;
		int64_t i = s, e2;
		int eq;
		
		if (s & 0x3) {
			e2 = (s & 0xfffffffffffffffcLL) + 4;
			for (; i < e2 && i < e; ++i) {
				if (v[i] == key) {
					return i;
				}
			}
		}
		xkey = _mm_set1_ps(key);
		e2 = (e & 0xfffffffffffffffcLL);
		for (; i < e2; i += 4) {
			xmm = _mm_load_ps(&v[i]);
			xmm = _mm_cmpeq_ps(xkey, xmm);
			eq = _mm_movemask_ps(xmm);
			if (eq != 0) {
				if (eq & 0x03) {
					if (eq & 0x1) {
						return i;
					} else {
						return i + 1;
					}
				} else {
					if (eq & 0x4) {
						return i + 2;
					} else {
						return i + 3;
					}
				}
			}
		}
		for (;i < e; ++i) {
			if (v[i] == key) {
				return i;
			}
		}
		return -1;
	}
#else
	{
		int64_t i;
		for (i = s; i < e; ++i) {
			if (v[i] == key) {
				return i;
			}
		}
		return -1;
	}
#endif
}


void 
nv_vector_mulmtr(nv_matrix_t *vec0, int m0,
				 const nv_matrix_t *vec1, int m1,
				 const nv_matrix_t *mat)
{
	int i;
	
	NV_ASSERT(vec0->n == vec1->n);
	NV_ASSERT(vec1->n == mat->n);
	NV_ASSERT(vec0->n == mat->m);
	
	for (i = 0; i < mat->m; ++i) {
		NV_MAT_V(vec0, m0, i) = nv_vector_dot(vec1, m1, mat, i);
	}
}

void
nv_vector_clip(nv_matrix_t *v, int v_j, float vmin, float vmax)
{
	int i;
	for (i = 0; i < v->n; ++i) {
		if (NV_MAT_V(v, v_j, i) < vmin) {
			NV_MAT_V(v, v_j, i) = vmin;
		}
		if (NV_MAT_V(v, v_j, i) > vmax) {
			NV_MAT_V(v, v_j, i) = vmax;
		}
	}
}
