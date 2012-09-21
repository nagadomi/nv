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

#ifndef NV_PSTABLE_H
#define NV_PSTABLE_H

#ifdef __cplusplus
extern "C" {
#endif


#include "nv_core.h"

typedef struct {
	float r;         /* 1G中のハッシュ関数の数,ハッシュの次元 */
	int k;           /* 1G中のハッシュ関数の数,ハッシュの次元 */
	int l;           /*ハッシュ関数Gの数 */
	nv_matrix_t *a;  /* 正規分布を要素に持つベクトル */
	nv_matrix_t *b;  /* 一様乱数 */
	nv_matrix_t *r1; /* ハッシュの整数値変換用の乱数 */
	nv_matrix_t *r2; /* ハッシュの整数値変換用の乱数 */
} nv_pstable_t;

typedef unsigned int nv_pstable_hash_t; /* 32bit unsigned int*/

nv_pstable_t *nv_pstable_alloc(int vector_dim);
nv_pstable_t *nv_pstable_alloc_ex(int vector_dim, int l, int k, float r);
void nv_pstable_hash(const nv_pstable_t *ps, nv_pstable_hash_t *hash, const nv_matrix_t *vec, int vec_m);

void nv_pstable_free(nv_pstable_t **pstable);

#ifdef __cplusplus
}
#endif


#endif

