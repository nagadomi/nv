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

/* 実験的な実装 */
/* 使っていない */

#include "nv_core.h"
#include "nv_num.h"
#if 0
#define NV_PSTABLE_L 1 /* L個のG(ハッシュ関数のセット) */
#define NV_PSTABLE_K 3 /* 1G中のハッシュ関数の数,ハッシュの次元 */
#define NV_PSTABLE_R 0.5f  /* ハッシュの衝突率を制御するパラメータ */

#define NV_PSTABLE_HASH_MAX  65535.0f

nv_pstable_t *
nv_pstable_alloc(int vector_dim)
{
	return nv_pstable_alloc_ex(vector_dim, NV_PSTABLE_L, NV_PSTABLE_K, NV_PSTABLE_R);
}

nv_pstable_t *
nv_pstable_alloc_ex(int vector_dim, int l, int k, float r)
{
	nv_pstable_t *ps = (nv_pstable_t *)nv_malloc(sizeof(nv_pstable_t));
	int m, n;

	memset(ps, 0, sizeof(*ps));

	ps->l = l;
	ps->k = k;
	ps->r = r;

	ps->a = nv_matrix3d_alloc(vector_dim, ps->l, ps->k);
	ps->b = nv_matrix3d_alloc(1, ps->l, ps->k);
	ps->r1 = nv_matrix3d_alloc(1, ps->l, ps->k);
	ps->r2 = nv_matrix3d_alloc(1, ps->l, ps->k);

	for (m = 0; m < ps->a->m; ++m) {
		for (n = 0; n < ps->a->n; ++n) {
			NV_MAT_V(ps->a, m, n) = nv_gaussian_rand(0.0f, 1.0f);
		}
	}
	for (m = 0; m < ps->b->m; ++m) {
		for (n = 0; n < ps->b->n; ++n) {
			NV_MAT_V(ps->b, m, n) = nv_rand() * ps->r;
		}
	}
	for (m = 0; m < ps->r1->m; ++m) {
		for (n = 0; n < ps->r1->n; ++n) {
			NV_MAT_V(ps->r1, m, n) = NV_FLOOR(
				nv_rand() * NV_PSTABLE_HASH_MAX * 2.0f - NV_PSTABLE_HASH_MAX
			);
			NV_MAT_V(ps->r2, m, n) = NV_FLOOR(
				nv_rand() * NV_PSTABLE_HASH_MAX * 2.0f - NV_PSTABLE_HASH_MAX
			);
		}
	}

	return ps;
}


void 
nv_pstable_free(nv_pstable_t **ps)
{
	if (ps != NULL && *ps != NULL) {
		nv_matrix_free(&(*ps)->a);
		nv_matrix_free(&(*ps)->b);
		nv_matrix_free(&(*ps)->r1);
		nv_matrix_free(&(*ps)->r2);
		nv_free(*ps);
		*ps = NULL;
	}
}

void 
nv_pstable_hash(const nv_pstable_t *ps, nv_pstable_hash_t *hash, const nv_matrix_t *vec, int vec_m)
{
	int k, l;
	float r_fac = 1.0f / ps->r;

	NV_ASSERT(ps->a->n == vec->n);

	memset(hash, 0, sizeof(nv_pstable_hash_t) * ps->l);

	for (l = 0; l < ps->l; ++l) {
		for (k = 0; k < ps->k; ++k) {
			/* s = (ax + b) / r */
			float s = nv_vector_dot(ps->a, NV_MAT_M(ps->a, l, k), vec, vec_m);
			s += NV_MAT3D_V(ps->b, l, k, 0);
			s *= r_fac;
			hash[l] = hash[l] * 33 + (int)s;
		}
	}
}

#endif

