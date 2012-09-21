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

#ifndef NV_ML_KMEANS_TREE_H
#define NV_ML_KMEANS_TREE_H

#ifdef __cplusplus
extern "C" {
#endif
	
#include "nv_core.h"

typedef struct {
	int height;
	int n;
	int m;
	int *dim;
	int *node;
	nv_matrix_t ***mat;
} nv_kmeans_tree_t;

nv_kmeans_tree_t *nv_kmeans_tree_alloc(int n, const int *width, int height);

void nv_kmeans_tree_train(nv_kmeans_tree_t *tree,
						 const nv_matrix_t *data,
						  int max_epoch);
void nv_kmeans_tree_inherit_train(nv_kmeans_tree_t *tree,
								  const nv_kmeans_tree_t *base_tree,
								  const nv_matrix_t *data,
								  int max_epoch);
void nv_kmeans_tree_train_at(nv_kmeans_tree_t *tree,
							 const nv_matrix_t *data,
							 int y, int x,
							 int max_epoch);

int nv_kmeans_tree_predict_label(const nv_kmeans_tree_t *tree,
								 const nv_matrix_t *vec, int vec_j);

int nv_kmeans_tree_predict_label_ex(const nv_kmeans_tree_t *tree,
									int height,
									const nv_matrix_t *vec, int vec_j,
									int nn);

nv_int_float_t nv_kmeans_tree_predict_label_and_dist_ex(const nv_kmeans_tree_t *tree,
														int height,
														const nv_matrix_t *vec, int vec_j,
														int nn);
nv_int_float_t nv_kmeans_tree_predict_label_and_dist(const nv_kmeans_tree_t *tree,
													 const nv_matrix_t *vec, int vec_j);
	
void nv_kmeans_tree_label_node(const nv_kmeans_tree_t *tree,
							   nv_matrix_t **mat, int *vec_j,
							   int label);

void nv_kmeans_tree_free(nv_kmeans_tree_t **tree);
void nv_kmeans_tree_progress(int flag);
void nv_kmeans_tree_dump_c(FILE *out,
						   const nv_kmeans_tree_t *tree,
						   const char *name, int static_variable);
void
nv_kmeans_tree_label_vec(const nv_kmeans_tree_t *tree,
						 nv_matrix_t **mat, int *vec_j,
						 int label);

#ifdef __cplusplus
}
#endif

#endif
