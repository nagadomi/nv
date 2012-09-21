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

#ifndef __NUM_SIMHASH_H
#define __NUM_SIMHASH_H
#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	nv_matrix_t *a;
} nv_simhash_t;

typedef uint64_t nv_simhash_hash_t;

nv_simhash_t *nv_simhash_alloc(int vector_dim);
void nv_simhash_free(nv_simhash_t **sh);
void nv_simhash_seed(nv_simhash_t *sh, nv_matrix_t *seed);
nv_simhash_hash_t nv_simhash_hash(const nv_simhash_t *sh, nv_matrix_t *vec, int vec_m);

void
nv_simhash_knn(const nv_simhash_t *sh,
			   const nv_simhash_hash_t *db, int64_t nhash, 
			   const nv_simhash_hash_t hash,
			   int64_t *results, int k);


#ifdef __cplusplus
}
#endif

#endif
