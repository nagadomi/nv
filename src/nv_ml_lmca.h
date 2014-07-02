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
#ifndef NV_ML_LMCA_H
#define NV_ML_LMCA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	NV_LMCA_DIAG, /* only diagonal element */
	NV_LMCA_FULL  /* full matrix */
} nv_lmca_type_e;

void nv_lmca_progress(int onoff);
void nv_lmca(nv_matrix_t *ldm,
			 const nv_matrix_t *data,
			 const nv_matrix_t *labels,
			 int nk, int mk, float margin, float pull_ratio, float delta,
			 int max_epoch);

void nv_lmca_init_cov(nv_matrix_t *ldm,
				 const nv_matrix_t *data);
void nv_lmca_init_random_projection(nv_matrix_t *ldm);
void nv_lmca_init_pca(nv_matrix_t *ldm,
					  const nv_matrix_t *data);
void nv_lmca_init_diag1(nv_matrix_t *ldm);
void nv_lmca_train(nv_matrix_t *ldm,
				   const nv_matrix_t *data, const nv_matrix_t *labels,
				   int nk, int mk,
				   float margin, float push_ratio, float delta,
				   int max_epoch);
void
nv_lmca_train_ex(nv_matrix_t *ldm,
				 nv_lmca_type_e type,
				 const nv_matrix_t *data, const nv_matrix_t *labels,
				 int nk, int mk,
				 float margin, float push_ratio, float delta,
				 int max_epoch);

void nv_lmca_projection(nv_matrix_t *v1, int v1_j,
						const nv_matrix_t *ldm,
						const nv_matrix_t *v2, int v2_j);
void nv_lmca_projection_all(nv_matrix_t *y,
							const nv_matrix_t *ldm,
							const nv_matrix_t *x);


#ifdef __cplusplus
}
#endif

#endif
