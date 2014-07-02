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

#ifndef NV_MATRIX_H
#define NV_MATRIX_H
#include "nv_config.h"
#include "nv_portable.h"

/* TODO: nv_matrix_list*系に対応していないところがある */

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	int list;
	int64_t list_step;
	int n;
	int m;
	int rows;
	int cols;
	int step;
	int alias;
	float *v;
} nv_matrix_t;

int64_t nv_vec_size(const nv_matrix_t *mat);
int64_t nv_mat_size(const nv_matrix_t *mat);
int64_t nv_mat_col(const nv_matrix_t *mat, int m);
int64_t nv_mat_row(const nv_matrix_t *mat, int m);
int64_t nv_mat_m(const nv_matrix_t *mat, int row, int col);
float *nv_mat3d_v(const nv_matrix_t *mat, int row, int col, int n);
float *nv_mat_v(const nv_matrix_t *mat, int m, int n);
float *nv_mat3d_list_v(const nv_matrix_t *mat, int list, int row, int col, int n);
float *nv_mat_list_v(const nv_matrix_t *mat, int list, int m, int n);

#define NV_MAT_VI(mat, m, n) ((int)NV_MAT_V(mat, m, n))
#define NV_MAT3D_VI(mat, row, col, n) ((int)NV_MAT3D_V(mat, row, col, n))

#if NV_ENABLE_STRICT
#  define NV_VEC_SIZE(mat) nv_vec_size(mat)
#  define NV_MAT_SIZE(mat) nv_mat_size(mat)
#  define NV_MAT_COL(mat, m) nv_mat_col(mat, m)
#  define NV_MAT_ROW(mat, m) nv_mat_row(mat, m)
#  define NV_MAT_M(mat, row, col) nv_mat_m(mat, row, col)
#  define NV_MAT3D_IDX(mat, row, col, n ) \
	(NV_MAT_M((mat), (row), (col)) * (mat)->step + (n))
#  define NV_MAT3D_V(mat, row, col, n) (*nv_mat3d_v(mat, row, col, n))
#  define NV_MAT_IDX(mat, m, n) \
	((int64_t)(m) * (mat)->step + (n))
#  define NV_MAT_V(mat, m, n) (*nv_mat_v(mat, m, n))
#  define NV_MAT3D_LIST_V(mat, list, row, col, n) (*nv_mat3d_list_v(mat, list, row, col, n))
#  define NV_MAT_LIST_V(mat, list, m, n) (*nv_mat_list_v(mat, list, m, n))
#else
#  define NV_VEC_SIZE(mat) \
	(sizeof(float) * (mat)->step)
#  define NV_MAT_SIZE(mat) \
	(NV_VEC_SIZE(mat) * (mat)->m)
#  define NV_MAT_COL(mat, m) \
	((int64_t)(m) % (mat)->cols)
#  define NV_MAT_ROW(mat, m) \
	(((int64_t)m) / (mat)->cols)
#  define NV_MAT_M(mat, row, col) \
	((int64_t)(mat)->cols * (row) + (col))
#  define NV_MAT3D_IDX(mat, row, col, n ) \
	(NV_MAT_M((mat), (row), (col)) * (mat)->step + (n))
#  define NV_MAT3D_V(mat, row, col, n) \
	((mat)->v[NV_MAT3D_IDX(mat, row, col, n)])
#  define NV_MAT_IDX(mat, m, n) \
	((int64_t)(m) * (mat)->step + (n))
#  define NV_MAT_V(mat, m, n) \
	((mat)->v[NV_MAT_IDX(mat, m, n)])
#  define NV_MAT3D_LIST_V(mat, list, row, col, n) \
	((mat)->v[(mat->list_step * list) + NV_MAT_M((mat), (row), (col)) * (mat)->step + (n)])
#  define NV_MAT_LIST_V(mat, list, m, n) \
	((mat)->v[((mat)->list_step * (list)) + (m) * (mat)->step + (n)])
#endif

/* 転置アクセス */

#define NV_MAT_VT(mat, m, n) NV_MAT_V(mat, n, m)

/* matrix */
nv_matrix_t *nv_matrix_alloc(int n, int m);
nv_matrix_t *nv_matrix_dup(const nv_matrix_t *base);
nv_matrix_t *nv_matrix_clone(const nv_matrix_t *base);
nv_matrix_t *nv_matrix_realloc(nv_matrix_t *oldmat, int new_m);
nv_matrix_t *nv_matrix_list_alloc(int n, int m, int list);
nv_matrix_t *nv_matrix3d_alloc(int n, int rows, int cols);
nv_matrix_t *nv_matrix3d_list_alloc(int n, int rows, int cols, int list);
nv_matrix_t *nv_matrix_alias(const nv_matrix_t *parent, int sn, int sm, int n, int m);
nv_matrix_t *nv_matrix_list_get(const nv_matrix_t *parent, int list);
void nv_matrix_zero(nv_matrix_t *mat);
void nv_vector_zero(nv_matrix_t *mat, int m);
void nv_matrix_free(nv_matrix_t **matrix);
void nv_vector_copy(nv_matrix_t *dest, int dm, const nv_matrix_t *src, int sm);
void nv_matrix_copy(nv_matrix_t *dest, int dm, const nv_matrix_t *src, int sm, int count_m);
void nv_matrix_copy_all(nv_matrix_t *dest, const nv_matrix_t *src);
void nv_matrix_m(nv_matrix_t *mat, int m);
void nv_matrix_fill(nv_matrix_t *mat, const float v);
void nv_vector_fill(nv_matrix_t *mat, int m, float v);
void nv_matrix_scale(nv_matrix_t *mat, float factor);
void nv_matrix_split(nv_matrix_t *mat1, int dest_n, const nv_matrix_t *mat2, int src_n);

void nv_matrix_print(FILE *out, const nv_matrix_t *mat);
void nv_vector_print(FILE *out, const nv_matrix_t *mat, int j);
void nv_matrix3d_print(FILE *out, const nv_matrix_t *mat, int channel);

void nv_matrix_dump_c(FILE *out, const nv_matrix_t *mat, const char *name, int static_variable);

void nv_vector_reshape(nv_matrix_t *mat,
					   const nv_matrix_t *vec, int vec_j);
void nv_matrix_reshape_vec(nv_matrix_t *vec, int vec_j,
						   const nv_matrix_t *mat);

nv_matrix_t *
nv_vector_shallow_reshape(nv_matrix_t *vec, int vec_j,
						  int n, int m);
nv_matrix_t *
nv_vector_shallow_reshape3d(nv_matrix_t *vec, int vec_j,
							int n, int rows, int cols);

typedef enum 
{
	NV_SORT_DIR_ASC = 0,
	NV_SORT_DIR_DESC = 1
} nv_sort_dir_e;

void
nv_matrix_sort(nv_matrix_t *mat, int sort_column_n, nv_sort_dir_e dir);

#ifdef __cplusplus
}
#endif

#endif
