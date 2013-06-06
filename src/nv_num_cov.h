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

#ifndef NV_NUM_COV_H
#define NV_NUM_COV_H

#ifdef __cplusplus
extern "C" {
#endif

/* 分散共分散行列 */
typedef struct {
	int n;
	int data_m;
	nv_matrix_t *u;        /* 平均 */
	nv_matrix_t *cov;      /* 行分散行列 */
	nv_matrix_t *eigen_vec; /* 固有ベクトル */
	nv_matrix_t *eigen_val; /* 固有値 (大きい順) */
} nv_cov_t;

nv_cov_t *nv_cov_alloc(int n);
void nv_cov(nv_matrix_t *cov,
			nv_matrix_t *u,
			const nv_matrix_t *data);
void
nv_cov_eigen_ex(nv_cov_t *cov, const nv_matrix_t *data, int eigen_n, int max_epoch);

void nv_cov_eigen(nv_cov_t *cov, const nv_matrix_t *data);
void nv_cov_free(nv_cov_t **cov);

#ifdef __cplusplus
}
#endif


#endif
