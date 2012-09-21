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
#include "mt64.h"
#include <time.h>

void 
nv_srand_time(void)
{
	unsigned long long seed = (unsigned long long)time(NULL);
	mt_init_genrand64(seed);
}

void 
nv_srand(unsigned int seed)
{
	mt_init_genrand64(seed);
}

float 
nv_rand(void)
{
	return (float)mt_genrand64_real1();
}

float 
nv_gaussian_rand(float average, float variance)
{
	float a = nv_rand();
	float b = nv_rand();
	return (sqrtf(-2.0f * logf(a)) * sinf(2.0f * NV_PI * b)) * variance + average;
}

int
nv_rand_index(int n)
{
	return (int)(n * mt_genrand64_real2());
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


