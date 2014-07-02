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

#include "nv_core.h"
#include "nv_core_matrix.h"

nv_matrix_t *
nv_matrix_clone(const nv_matrix_t *base)
{
	return nv_matrix3d_list_alloc(base->n, base->rows, base->cols, base->list);
}

nv_matrix_t *
nv_matrix_dup(const nv_matrix_t *base)
{
	nv_matrix_t *mat = nv_matrix_clone(base);
	if (mat != NULL) {
		nv_matrix_copy_all(mat, base);
	}
	return mat;
}

nv_matrix_t *
nv_matrix_realloc(nv_matrix_t *mat, int new_m)
{
	size_t mem_size = (size_t)mat->step * new_m * sizeof(float);
	float *v = (float *)nv_aligned_realloc(mat->v, 0x20, mem_size);

	NV_ASSERT(mat->rows == 1);
	NV_ASSERT(mat->list == 1);
	
	mat->m = new_m;
	mat->rows = 1;
	mat->cols = new_m;
	mat->list_step = (int64_t)mat->step * mat->m;
	mat->v = v;
	
	return mat;
}

static nv_matrix_t *
nv_matrix_list_alloc32(int n, int m, int list)
{
	size_t step;
	int step_n;
	size_t mem_size;
	nv_matrix_t *matrix;
	void *p;
	
	NV_ASSERT(n != 0 && m != 0);
	if (n > 4 && n % 8 != 0) {	
		// SSE ALIGN
		step_n = (n + 8 - (n & 7));
	} else {
		step_n = n;
	}
	step = step_n * sizeof(float);

	matrix = nv_alloc_type(nv_matrix_t, 1);
	mem_size = ((size_t)step * m) * list;
	if (nv_aligned_malloc(&p, 0x20, mem_size) != 0) {
		matrix->v = NULL;
		return NULL;
	}
	matrix->v = (float *)p;
	matrix->n = n;
	matrix->m = m;
	matrix->step = step_n;
	matrix->rows = 1;
	matrix->cols = m;
	matrix->list = list;
	matrix->list_step = (int64_t)matrix->step * matrix->m;
	matrix->alias = 0;

	return matrix;
}

nv_matrix_t *
nv_matrix_list_alloc(int n, int m, int list)
{
	return nv_matrix_list_alloc32(n, m, list);
}

nv_matrix_t *
nv_matrix_alloc(int n, int m)
{
	return nv_matrix_list_alloc32(n, m, 1);
}

nv_matrix_t *
nv_matrix3d_alloc(int n, int rows, int cols)
{
	nv_matrix_t *mat = nv_matrix_alloc(n, rows * cols);
	mat->rows = rows;
	mat->cols = cols;

	return mat;
}

nv_matrix_t *
nv_matrix3d_list_alloc(int n, int rows, int cols, int list)
{
	nv_matrix_t *mat = nv_matrix_list_alloc(n, rows * cols, list);
	mat->rows = rows;
	mat->cols = cols;

	return mat;
}

nv_matrix_t *
nv_matrix_alias(const nv_matrix_t *parent, int sn, int sm, int n, int m)
{
	nv_matrix_t *matrix = (nv_matrix_t *)nv_malloc(sizeof(nv_matrix_t));
	matrix->list = 1;
	matrix->n = n;
	matrix->m = m;
	matrix->rows = 1;
	matrix->cols = m;
	matrix->v = &parent->v[sm * parent->step + sn];
	matrix->step = parent->step;
	matrix->list_step = matrix->step * matrix->m;
	matrix->alias = 1;

	return matrix;
}

nv_matrix_t *
nv_matrix_list_get(const nv_matrix_t *parent, int list)
{
	nv_matrix_t *matrix = (nv_matrix_t *)nv_malloc(sizeof(nv_matrix_t));
	matrix->list = 1;
	matrix->n = parent->n;
	matrix->m = parent->m;
	matrix->rows = parent->rows;
	matrix->cols = parent->cols;
	matrix->v = &NV_MAT_LIST_V(parent, list, 0, 0);
	matrix->step = parent->step;
	matrix->list_step = parent->list_step;
	matrix->alias = 1;

	return matrix;
}

void
nv_matrix_zero(nv_matrix_t *mat)
{
	memset(mat->v, 0, (size_t)mat->list_step * mat->list * sizeof(float));
}

void
nv_vector_zero(nv_matrix_t *mat, int m)
{
	memset(&NV_MAT_V(mat, m, 0), 0, mat->step * sizeof(float));
}

void
nv_matrix_copy(nv_matrix_t *dest, int dm, const nv_matrix_t *src, int sm, int count_m)
{
	NV_ASSERT(dest->n == src->n);
	memmove(&NV_MAT_V(dest, dm, 0), &NV_MAT_V(src, sm, 0), dest->step * count_m * sizeof(float));
}

void nv_matrix_copy_all(nv_matrix_t *dest, const nv_matrix_t *src)
{
	NV_ASSERT(dest->n == src->n);
	NV_ASSERT(dest->m >= src->m);

	memmove(dest->v, src->v, (size_t)src->list_step * src->list * sizeof(float));
}
void
nv_matrix_free(nv_matrix_t **matrix)
{
	if (*matrix != NULL) {
		if ((*matrix)->alias == 0) {
			nv_aligned_free((*matrix)->v);
		}
		nv_free(*matrix);
		*matrix = NULL;
	}
}

void
nv_vector_print(FILE *out, const nv_matrix_t *mat, int j)
{
	int n;

	fprintf(out, "\t[ ");
	for (n = 0; n < mat->n; ++n) {
		if (n != 0) {
			fprintf(out, ", ");
		}
		fprintf(out, "%10E", NV_MAT_V(mat, j, n));
	}
	fprintf(out, "]\n");
}

void
nv_matrix_print(FILE *out, const nv_matrix_t *mat)
{
	int m, n;

	fprintf(out, "(");
	for (m = 0; m < mat->m; ++m) {
		if (m != 0) {
			fprintf(out, ",\n");
		}
		fprintf(out, "\t[ ");
		for (n = 0; n < mat->n; ++n) {
			if (n != 0) {
				fprintf(out, ", ");
			}
			fprintf(out, "%10E", NV_MAT_V(mat, m, n));
		}
		fprintf(out, "]");
	}
	fprintf(out, ");\n");
}

void
nv_matrix3d_print(FILE *out, const nv_matrix_t *mat, int channel)
{
	int row, col;

	fprintf(out, "(");
	for (row = 0; row < mat->rows; ++row) {
		if (row != 0) {
			fprintf(out, ",\n");
		}
		fprintf(out, "[ ");
		for (col = 0; col < mat->cols; ++col) {
			if (col != 0) {
				fprintf(out, ", ");
			}
			fprintf(out, "%10E", NV_MAT3D_V(mat, row, col, channel));
		}
		fprintf(out, "]");
	}
	fprintf(out, ");\n");
}

void
nv_matrix_dump_c(FILE *out, const nv_matrix_t *mat, const char *name, int static_variable)
{
	int i, m, start_flag = 1;

	fprintf(out, "NV_ALIGNED(static float, %s_v[%d], 16) = {\n", name, mat->m * mat->step);
	for (m = 0; m < mat->m; ++m) {
		for (i = 0; i < mat->n; ++i) {
			if (!start_flag) {
				fprintf(out, ",");
				if (i != 0 && i % 5 == 0) {
					fprintf(out, "\n");
				}
			} else {
				start_flag = 0;
			}
			fprintf(out, "%15Ef", NV_MAT_V(mat, m, i));
		}
		for (i = mat->n; i < mat->step; ++i) {
			if (!start_flag) {
				fprintf(out, ",");
				if (i != 0 && i % 5 == 0) {
					fprintf(out, "\n");
				}
			} else {
				start_flag = 0;
			}
			fprintf(out, "%15Ef", 0.0f);
		}
		fprintf(out, "\n");
	}
	fprintf(out, "};\n");
	fprintf(out, "%snv_matrix_t %s = {\n %d, %"PRId64", %d, %d, %d, %d, %d, %d, %s_v\n};\n",
		static_variable ? "static ":"",
		name, mat->list, mat->list_step, mat->n, mat->m, mat->rows, mat->cols, mat->step, 0,
		name);
	fflush(out);
}

void
nv_vector_copy(nv_matrix_t *dest, int dm, const nv_matrix_t *src, int sm)
{
	NV_ASSERT(dest->n >= src->n);

	memmove(&NV_MAT_V(dest, dm, 0), &NV_MAT_V(src, sm, 0), src->step * sizeof(float));
}

void
nv_matrix_m(nv_matrix_t *mat, int m)
{
	if (mat->rows == 1) {
		mat->cols = m;
	} else {
		int diff = mat->m - m;
		if (diff % mat->cols) {
			mat->rows -= (mat->m - m) / mat->cols; 
		} else {
			mat->rows -= (mat->m - m) / mat->cols;
			if (diff > 0) {
				--mat->rows;
			} else {
				++mat->rows;
			}
		}
	}
	mat->m = m;
	mat->list_step = (int64_t)mat->step * mat->m;	
}

void 
nv_vector_fill(nv_matrix_t *mat, int m, float v)
{
	if (mat->n < 64) {
		int i;
		for (i = 0; i < mat->n; ++i) {
			NV_MAT_V(mat, m, i) = v;
		}
	} else {
		int j, k;
		float *p = &NV_MAT_V(mat, m, 0);
		p[0] = v;
		k = 1;
		j = 1;
		while (k * 2 < mat->n) {
			memmove(&p[j], &p[0], k * sizeof(float));
			j += k;
			k *= 2;
		}
		if (k > 1) {
			k /= 2;
		}
		while (j < mat->n) {
			if (j + k <= mat->n) {
				memmove(&p[j], &p[0], k * sizeof(float));
				j += k;
			} else {
				if (k > 1) {
					k /= 2;
				}
			}
		}
	}
}

void 
nv_matrix_fill(nv_matrix_t *mat, const float v)
{
	int64_t j, k;
	
	if (mat->m == 0) {
		return;
	} else {
		mat->v[0] = v;
		k = 1;
		j = 1;
		while (k * 2 < mat->list_step) {
			memmove(&mat->v[j], &mat->v[0], k * sizeof(float));
			j += k;
			k *= 2;
		}
		if (k > 1) {
			k /= 2;
		}
		while (j < mat->list_step) {
			if (j + k <= mat->list_step) {
				memmove(&mat->v[j], &mat->v[0], k * sizeof(float));
				j += k;
			} else {
				if (k > 1) {
					k /= 2;
				}
			}
		}
	}
}

void
nv_matrix_scale(nv_matrix_t *mat, float factor)
{
	int m, n;

	for (m = 0; m < mat->m; ++m) {
		for (n = 0; n < mat->n; ++n) {
			NV_MAT_V(mat, m, n) *= factor;
		}
	}
}

void 
nv_matrix_split(nv_matrix_t *mat1, int dest_n, const nv_matrix_t *mat2, int src_n)
{
	int m;

	NV_ASSERT(mat1->m == mat2->m);

	for (m = 0; m < mat1->m; ++m) {
		NV_MAT_V(mat1, m, dest_n) = NV_MAT_V(mat2, m, src_n);
	}
}

static int
nv_column_cmp_asc(const void *p1, const void *p2)
{
	const float f1 = *(const float *)p1;
	const float f2 = *(const float *)p2;

	if (f1 > f2) {
		return 1;
	} else if (f1 < f2) {
		return -1;
	}
	return 0;
}

static int
nv_column_cmp_desc(const void *p1, const void *p2)
{
	const float f1 = *(const float *)p1;
	const float f2 = *(const float *)p2;

	if (f1 < f2) {
		return 1;
	} else if (f1 > f2) {
		return -1;
	}
	return 0;
}

void
nv_matrix_sort(nv_matrix_t *mat, int sort_column_n, nv_sort_dir_e dir)
{
	nv_matrix_t *sort_data = nv_matrix_alloc(2, mat->m);
	nv_matrix_t *tmp = nv_matrix_alloc(mat->n, mat->m);
	int m;

	for (m = 0; m < mat->m; ++m) {
		NV_MAT_V(sort_data, m, 0) = NV_MAT_V(mat, m, sort_column_n);
		NV_MAT_V(sort_data, m, 1) = (float)m;
	}
	if (dir == NV_SORT_DIR_ASC) {
		qsort(sort_data->v, sort_data->m,
			sort_data->step * sizeof(float), nv_column_cmp_asc);
	} else {
		qsort(sort_data->v, sort_data->m,
			sort_data->step * sizeof(float), nv_column_cmp_desc);
	}
	for (m = 0; m < mat->m; ++m) {
		nv_vector_copy(tmp, m, mat, (int)NV_MAT_V(sort_data, m, 1));
	}
	nv_matrix_copy(mat, 0, tmp, 0, mat->m);

	nv_matrix_free(&sort_data);
	nv_matrix_free(&tmp);
}

int64_t
nv_vec_size(const nv_matrix_t *mat)
{
	return sizeof(float) * mat->step;
}

int64_t
nv_mat_size(const nv_matrix_t *mat)
{
	return (int64_t)nv_vec_size(mat) * mat->m;
}

int64_t
nv_mat_col(const nv_matrix_t *mat, int m)
{
	NV_ASSERT(mat->m > m);
	return (int64_t)m % mat->cols;
}

int64_t
nv_mat_row(const nv_matrix_t *mat, int m)
{
	NV_ASSERT(mat->m > m);
	return (int64_t)m / mat->cols;
}

int64_t
nv_mat_m(const nv_matrix_t *mat, int row, int col)
{
	NV_ASSERT(mat->rows > row);
	NV_ASSERT(mat->cols > col);
	return (int64_t)mat->cols * row + col;
}

float *
nv_mat3d_v(const nv_matrix_t *mat, int row, int col, int n)
{
	NV_ASSERT(mat->rows > row);
	NV_ASSERT(mat->cols > col);
	NV_ASSERT(mat->n > n);
	NV_ASSERT(mat->m > nv_mat_m(mat, row, col));
	NV_ASSERT(n >= 0);
	NV_ASSERT(row >= 0);
	NV_ASSERT(col >= 0);
	
	return &mat->v[nv_mat_m(mat, row, col) * mat->step + n];
}

float *
nv_mat_v(const nv_matrix_t *mat, int m, int n)
{
	NV_ASSERT(mat->m > m);
	NV_ASSERT(mat->n > n);
	NV_ASSERT(m >= 0);
	NV_ASSERT(n >= 0);
	return &mat->v[(int64_t)m * mat->step + n];
}

float *
nv_mat3d_list_v(const nv_matrix_t *mat, int list, int row, int col, int n)
{
	NV_ASSERT(mat->list > list);
	NV_ASSERT(mat->rows > row);
	NV_ASSERT(mat->cols > col);
	NV_ASSERT(mat->n > n);
	NV_ASSERT(n >= 0);
	NV_ASSERT(row >= 0);
	NV_ASSERT(col >= 0);
	NV_ASSERT(list >= 0);
	
	return &mat->v[mat->list_step * list + nv_mat_m(mat, row, col) * mat->step + n];
}

float *
nv_mat_list_v(const nv_matrix_t *mat, int list, int m, int n)
{
	NV_ASSERT(mat->list > list);
	NV_ASSERT(mat->m > m);
	NV_ASSERT(mat->n > n);
	NV_ASSERT(n >= 0);
	NV_ASSERT(m >= 0);
	
	return &mat->v[mat->list_step * list + m * mat->step + n];
}

/* shallow copy */
nv_matrix_t *
nv_vector_shallow_reshape(nv_matrix_t *vec, int vec_j,
						  int n, int m)
{
	nv_matrix_t *mat;

	NV_ASSERT(vec->n == n * m);
	NV_ASSERT(n < 8 || n % 8 == 0); // n >= 8 && n % 8 != 0, AVX does not work	

	if (!(vec->n == n * m && (n < 8 || n % 8 == 0))) {
		return NULL;
	}
	mat = (nv_matrix_t *)nv_malloc(sizeof(nv_matrix_t));
	mat->list = 1;
	mat->n = n;
	mat->m = m;
	mat->rows = 1;
	mat->cols = m;
	mat->v = &NV_MAT_V(vec, vec_j, 0);
	mat->step = n;
	mat->list_step = (int64_t)mat->step * mat->m;
	mat->alias = 1;
	
	return mat;
}

nv_matrix_t *
nv_vector_shallow_reshape3d(nv_matrix_t *vec, int vec_j,
							int n, int rows, int cols)
{
	nv_matrix_t *mat;

	NV_ASSERT(vec->n == n * rows * cols);
	NV_ASSERT(n < 8 || n % 8 == 0); // n >= 8 && n % 8 != 0, AVX does not work
	
	if (!(vec->n == n * rows * cols && (n < 8 || n % 8 == 0))) {
		return NULL;
	}
	mat = (nv_matrix_t *)nv_malloc(sizeof(nv_matrix_t));
	mat->list = 1;
	mat->n = n;
	mat->m = rows * cols;
	mat->rows = rows;
	mat->cols = cols;
	mat->v = &NV_MAT_V(vec, vec_j, 0);
	mat->step = n;
	mat->list_step = (int64_t)mat->step * mat->m;
	mat->alias = 1;
	
	return mat;
}

void
nv_vector_reshape(nv_matrix_t *mat,
				  const nv_matrix_t *vec, int vec_j)
{
	int j;
	NV_ASSERT(mat->n * mat->m == vec->n);
	
	for (j = 0; j < mat->m; ++j) {
		memmove(&NV_MAT_V(mat, j, 0), &NV_MAT_V(vec, vec_j, mat->n * j),
				sizeof(float) * mat->n);
	}
}
// 0 1 2 3 4 5 6 7 8
// 0 1 2 | 3 4 5 | 6 7 8
// 


void
nv_matrix_reshape_vec(nv_matrix_t *vec, int vec_j,
					  const nv_matrix_t *mat)
{
	int j;
	NV_ASSERT(mat->n * mat->m == vec->n);
	
	for (j = 0; j < mat->m; ++j) {
		memmove(&NV_MAT_V(vec, vec_j, mat->n * j), &NV_MAT_V(mat, j, 0),
				sizeof(float) * mat->n);
	}
}
