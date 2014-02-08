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

#ifndef NV_NUM_MATRIX_H
#define NV_NUM_MATRIX_H

#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

nv_matrix_t *nv_matrix_tr(const nv_matrix_t *mat);
void nv_matrix_tr_ex(nv_matrix_t *tr, const nv_matrix_t *mat);

nv_matrix_t *nv_matrix3d_tr(const nv_matrix_t *mat);
void nv_matrix_muls(nv_matrix_t *y, const nv_matrix_t *a, float scale);
void nv_matrix_adds(nv_matrix_t *y, const nv_matrix_t *a, float val);
void nv_matrix_add(nv_matrix_t *y, const nv_matrix_t *a, const nv_matrix_t *b);
void nv_matrix_sub(nv_matrix_t *y, const nv_matrix_t *a, const nv_matrix_t *b);

void nv_matrix_var(nv_matrix_t *y, int y_j, const nv_matrix_t *data);
void nv_matrix_mean(nv_matrix_t *y, int y_j, const nv_matrix_t *data); 
void nv_matrix_var_ex(nv_matrix_t *y, int y_j, const nv_matrix_t *data,
					  const nv_matrix_t *mean, int mean_j);
void nv_matrix_normalize_shift(nv_matrix_t *mat, float min_v, float max_v);
	
typedef enum {
	NV_MAT_TR,
	NV_MAT_NOTR
} nv_matrix_tr_t;

// y = A * x
// y = A' * x
void nv_matrix_mulv(nv_matrix_t *y, int ym,
					const nv_matrix_t *a,
					nv_matrix_tr_t a_tr,
					const nv_matrix_t *x,
					int xm);

// Y = A  * X
// Y = A  * X'
// Y = A' * X
// Y = A' * X'
void nv_matrix_mul(nv_matrix_t *y,
				   const nv_matrix_t *a,
				   nv_matrix_tr_t a_tr,
				   const nv_matrix_t *b,
				   nv_matrix_tr_t b_tr);
void nv_matrix_diag(nv_matrix_t *diag,
					nv_matrix_t *vec,
					int vec_j);
	
#ifdef __cplusplus
}
#endif

#endif
