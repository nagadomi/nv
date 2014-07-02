/*
 * This file is part of libnv.
 *
 * Copyright (C) 2014 nagadomi@nurs.or.jp
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

#ifndef NV_NUM_ZCA_H
#define NV_NUM_ZCA_H

#ifdef __cplusplus
extern "C" {
#endif

void nv_zca_train(nv_matrix_t *mean, int mean_j,
				  nv_matrix_t *u,
				  const nv_matrix_t *a,
				  float epsilon);

void nv_zca_train_ex(nv_matrix_t *mean, int mean_j,
					 nv_matrix_t *u,
					 const nv_matrix_t *a,
					 float epsilon,
					 int npca,
					 int keep_space);

void nv_zca_whitening(nv_matrix_t *x, int x_j,
					  const nv_matrix_t *mean, int mean_j,
					  const nv_matrix_t *u);

void nv_zca_whitening_all(nv_matrix_t *a,
						  const nv_matrix_t *mean, int mean_j,
						  const nv_matrix_t *u);

#ifdef __cplusplus
}
#endif

#endif
