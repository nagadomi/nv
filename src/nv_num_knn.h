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

#ifndef NV_NUM_KNN_H
#define NV_NUM_KNN_H

#include "nv_core.h"
#include "nv_num_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	int index;
	float dist;
} nv_knn_result_t;

typedef float (*nv_knn_func_t)(const nv_matrix_t *vec1, int m1, const nv_matrix_t *vec2, int m2);

int nv_nn(const nv_matrix_t *mat, const nv_matrix_t *vec, int m);
int nv_nn_ex(const nv_matrix_t *mat,
			 const nv_matrix_t *vec, int vec_m,
			 nv_knn_func_t func);
nv_int_float_t
nv_nn_dist(const nv_matrix_t *mat,
		   const nv_matrix_t *vec, int vec_j);
	
int nv_knn(nv_knn_result_t *results, int n,
		   const nv_matrix_t *mat,
		   const nv_matrix_t *vec, int vec_m);
int nv_knn_ex(nv_knn_result_t *results, int n,
			  const nv_matrix_t *mat,
			  const nv_matrix_t *vec, int vec_m,
			  nv_knn_func_t func);

#ifdef __cplusplus
}
#endif
#endif

