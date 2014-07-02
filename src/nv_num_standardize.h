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

#ifndef NV_NUM_STANDARDIZE_H
#define NV_NUM_STANDARDIZE_H

#ifdef __cplusplus
extern "C" {
#endif

void nv_standardize(nv_matrix_t *x, int x_j,
					const nv_matrix_t *mean, int mean_j,
					const nv_matrix_t *sd, int sd_j);


void nv_standardize_local(nv_matrix_t *x, int x_j, float epsilon);

void nv_standardize_train(nv_matrix_t *mean, int mean_j,
						  nv_matrix_t *sd, int sd_j,
						  const nv_matrix_t *data,
						  float epsilon);

void nv_standardize_all(nv_matrix_t *a,
						const nv_matrix_t *mean, int mean_j,
						const nv_matrix_t *sd, int sd_j);

void nv_standardize_local_all(nv_matrix_t *a, float epsilon);

#ifdef __cplusplus
}
#endif

#endif
