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
#include "tinymt32.h"
#include <time.h>
#include <limits.h>

#define NV_RAND_THREAD_MAX 128

tinymt32_t g_state[128];

void
nv_rand_init(void)
{
	int i;
	for (i = 0; i < NV_RAND_THREAD_MAX; ++i) {
		tinymt32_init_param(&g_state[i],
							0x09f6013eU,
							0xf848fe13U,
							0x52a0f5ffU);
	}
	tinymt32_init(&g_state[0], 11);
	for (i = 1; i < NV_RAND_THREAD_MAX; ++i) {
		tinymt32_init(&g_state[i], 11 + i);
	}
}

void 
nv_srand_time(void)
{
	uint32_t seed = (uint32_t)(time(NULL) & 0xffffffff);
	int i;
	
	tinymt32_init(&g_state[0], seed);
	for (i = 1; i < NV_RAND_THREAD_MAX; ++i) {
		tinymt32_init(&g_state[i], seed + i);
	}
}

void 
nv_srand(unsigned int seed)
{
	int i;
	tinymt32_init(&g_state[0], seed);
	for (i = 1; i < NV_RAND_THREAD_MAX; ++i) {
		tinymt32_init(&g_state[i], seed + i);
	}
}

float 
nv_rand(void)
{
	int thread_id = nv_omp_thread_id();
	NV_ASSERT(thread_id < NV_RAND_THREAD_MAX);
	return tinymt32_generate_float01(&g_state[thread_id]);
}

float 
nv_nrand(float average, float variance)
{
	float a = nv_rand();
	float b = nv_rand();
	return (sqrtf(-2.0f * logf(a)) * sinf(2.0f * NV_PI * b)) * variance + average;
}

int
nv_rand_index(int n)
{
	int thread_id = nv_omp_thread_id();
	int index;
	
	NV_ASSERT(thread_id < NV_RAND_THREAD_MAX);
	
	index = (int)(n * tinymt32_generate_32double(&g_state[thread_id]));
	if (index == n) {
		index -= 1;
	}
	
	return index;
}

void 
nv_shuffle_index(int *a, int start, int end)
{
	int i, tmp;
	int n = end - start;
	int rnd;

	for (i = 0; i < n; ++i) {
		a[i] = i + start;
	}
	for (i = 1; i < n; ++i) {
		rnd = (int)(nv_rand() * i);
		if (rnd < n) {
			tmp = a[i];
			a[i] = a[rnd];
			a[rnd] = tmp;
		}
	}
}

void 
nv_vector_shuffle(nv_matrix_t *mat)
{
	int i, rnd;
	int n = mat->m;
	nv_matrix_t *tmp = nv_matrix_alloc(mat->n, 1);

	for (i = 1; i < n; ++i) {
		rnd = (int)(nv_rand() * i);
		if (rnd < n) {
			nv_vector_copy(tmp, 0, mat, i);
			nv_vector_copy(mat, i, mat, rnd);
			nv_vector_copy(mat, rnd, tmp, 0);
		}
	}

	nv_matrix_free(&tmp);
}

void 
nv_vector_shuffle_pair(nv_matrix_t *mat1, nv_matrix_t *mat2)
{
	int i, rnd;
	int n = mat1->m;
	nv_matrix_t *tmp1 = nv_matrix_alloc(mat1->n, 1);
	nv_matrix_t *tmp2 = nv_matrix_alloc(mat2->n, 1);

	for (i = 1; i < n; ++i) {
		rnd = (int)(nv_rand() * i);
		if (rnd < n) {
			nv_vector_copy(tmp1, 0, mat1, i);
			nv_vector_copy(mat1, i, mat1, rnd);
			nv_vector_copy(mat1, rnd, tmp1, 0);

			nv_vector_copy(tmp2, 0, mat2, i);
			nv_vector_copy(mat2, i, mat2, rnd);
			nv_vector_copy(mat2, rnd, tmp2, 0);
		}
	}
	nv_matrix_free(&tmp1);
	nv_matrix_free(&tmp2);
}

void 
nv_vector_rand(nv_matrix_t *v, int v_j, float rmin, float rmax)
{
	int n;
	for (n = 0; n < v->n; ++n) {
		NV_MAT_V(v, v_j, n) = nv_rand() * (rmax - rmin) + rmin;
	}
}

void 
nv_vector_nrand(nv_matrix_t *v, int v_j, float u, float s)
{
	int n;
	for (n = 0; n < v->n; ++n) {
		NV_MAT_V(v, v_j, n) = nv_nrand(u, s);
	}
}

void
nv_matrix_rand(nv_matrix_t *mat, float rmin, float rmax)
{
	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif	
	for (i = 0; i < mat->m; ++i) {
		nv_vector_rand(mat, i, rmin, rmax);
	}
}

void
nv_matrix_nrand(nv_matrix_t *mat, float u, float s)
{
	int i;
#ifdef _OPENMP
#pragma omp parallel for	
#endif	
	for (i = 0; i < mat->m; ++i) {
		nv_vector_nrand(mat, i, u, s);
	}
}
