/*
 * This file is part of libnv.
 *
 * Copyright (C) 2011 nagadomi@nurs.or.jp
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

#define NV_SIMHASH_BITS 64

nv_simhash_t *
nv_simhash_alloc(int vector_dim)
{
	nv_simhash_t *sh = (nv_simhash_t *)nv_malloc(sizeof(nv_simhash_t));
	int m, n;
	memset(sh, 0, sizeof(*sh));

	sh->a = nv_matrix_alloc(vector_dim, NV_SIMHASH_BITS);
	for (m = 0; m < sh->a->m; ++m) {
		for (n = 0; n < sh->a->n; ++n) {
			NV_MAT_V(sh->a, m, n) = nv_gaussian_rand(0.0f, 1.0f);
		}
	}
	return sh;
}

void 
nv_simhash_free(nv_simhash_t **sh)
{
	if (sh != NULL && *sh != NULL) {
		nv_matrix_free(&(*sh)->a);
		nv_free(*sh);
		*sh = NULL;
	}
}

void 
nv_simhash_seed(nv_simhash_t *sh, nv_matrix_t *seed)
{
	NV_ASSERT(sh->a->n == seed->n);
	NV_ASSERT(sh->a->m == seed->m);

	nv_matrix_copy(sh->a, 0, seed, 0, sh->a->m);
}

nv_simhash_hash_t
nv_simhash_hash(const nv_simhash_t *sh,
				nv_matrix_t *vec, int vec_j)
{
	nv_simhash_hash_t hash = 0;
	int64_t j;

	NV_ASSERT(sh->a->n == vec->n);
	NV_ASSERT(sh->a->m == NV_SIMHASH_BITS);

	for (j = 0; j < sh->a->m; ++j) {
		if (nv_vector_dot(sh->a, j, vec, vec_j) > 0.0f) {
			hash |= (1ULL << j);
		}
	}

	return hash;
}


void
nv_simhash_knn(const nv_simhash_t *sh,
			   const nv_simhash_hash_t *db, int64_t nhash, 
			   const nv_simhash_hash_t hash,
			   int64_t *results, int k)
{
	int64_t j;
	int i;
	nv_matrix_t *dists = nv_matrix_alloc(2, nhash);
	nv_matrix_zero(dists);

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (j = 0; j < nhash; ++j) {
		NV_MAT_V(dists, j, 0) = (float)(NV_POPCNT_U64(db[j] ^ hash));
		NV_MAT_V(dists, j, 1) = (float)j;
	}
	nv_matrix_sort(dists, 0, NV_SORT_DIR_ASC);
	for (i = 0; i < k; ++i) {
		results[i] = (int)NV_MAT_V(dists, i, 1);
	}

	nv_matrix_free(&dists);
}
