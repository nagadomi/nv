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
#include "nv_num_vector.h"
#include "nv_num_munkres.h"


/* 割当問題 */
/* 参考: http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html */

#define NV_MUNKRES_ZERO  0
#define NV_MUNKRES_STAR  1
#define NV_MUNKRES_PRIME 2

typedef struct {
	int n;
	int m;
	int **v;
} nv_munkres_mask_t;

static void
clear_covers(int *r_cov, int *c_cov, int n)
{
	memset(r_cov, 0, sizeof(int) * n);
	memset(c_cov, 0, sizeof(int) * n);
}

static int
step1(nv_matrix_t *mat)
{
	int j;
	
	for (j = 0; j < mat->m; ++j) {
		nv_vector_subs(mat, j, mat, j,
					   nv_vector_mins(mat, j));
	}
	
	return 2; /* 2へ */
}

static int
step2(const nv_matrix_t *mat,
	  nv_munkres_mask_t *mask,
	  int *r_cov, int *c_cov)
{
	int j, i;
	
	/* 行か列の0要素にスターをひとつつける */
	for (j = 0; j < mat->m; ++j) {
		if (r_cov[j] == 0) {
			for (i = 0; i < mat->n; ++i) {
				if (c_cov[i] == 0
					&& fabsf(NV_MAT_V(mat, j, i)) < FLT_EPSILON)
				{
					mask->v[j][i] = NV_MUNKRES_STAR;
					r_cov[j] = 1; /* 保護 */
					c_cov[i] = 1; /* 保護 */
				}
			}
		}
	}
	clear_covers(r_cov, c_cov, NV_MAX(mat->n, mat->m));
	
	return 3; /* 3へ */
}

static int
step3(const nv_munkres_mask_t *mask,
	  int *c_cov)
{
	int j, i, c = 0;

	/* 星の着いてる列を数える */
	for (j = 0; j < mask->m; ++j) {
		for (i = 0; i < mask->n; ++i) {
			if (mask->v[j][i] == NV_MUNKRES_STAR) {
				c_cov[i] = 1;
			}
		}
	}
	for (i = 0; i < mask->n; ++i) {
		if (c_cov[i] == 1) {
			++c;
		}
	}
	if (c >= mask->n || c >= mask->m) {
		return 0; /* すべてユニーク、終了 */
	}
	return 4; /* 4へ  */
}

static void
find_a_zero(const nv_matrix_t *mat,
			const int *r_cov,
			const int *c_cov,
			int *row, int *col)
{
	int j, i;
	
	*row = *col = -1;
	
	for (j = 0; j < mat->m; ++j) {
		if (r_cov[j] == 0) {
			for (i = 0; i < mat->n; ++i) {
				if (NV_MAT_V(mat, j, i) == 0.0f
					&& c_cov[i] == 0)
				{
					*row = j;
					*col = i;
				}
			}
			if (*row != -1) {
				return;
			}
		}
	}
}

static int
find_star_in_row(const nv_munkres_mask_t *mask,
				 const int row, int *col)
{
   int i = 0;
   
   *col = -1;
   for (i = 0; i < mask->n; ++i) {
	   if (mask->v[row][i] == NV_MUNKRES_STAR) {
		   *col = i;
	   }
   }
   
   return *col != -1 ? 1 : 0;
}

static int
step4(const nv_matrix_t *mat,
	  nv_munkres_mask_t *mask,
	  int *r_cov, int *c_cov,
	  int *z_0r, int *z_0c)
{
	int row = -1;
	int col = -1;
	int step = 0;
	int col2 = -1;

	do {
		/* 保護されていない0を探す  */
		find_a_zero(mat, r_cov, c_cov, &row, &col);
		if (row < 0) {
			/* ない */
			step = 6; /* 6へ */
		} else {
			mask->v[row][col] = NV_MUNKRES_PRIME;
			/* 行から星の着いてる列を探す */
			if (find_star_in_row(mask, row, &col2)) {
				col = col2;
				r_cov[row] = 1;
				c_cov[col] = 0;
			} else {
				/* ない */
				*z_0r = row;
				*z_0c = col;
				step = 5; /* 5へ */
			}
		}
	} while (!step);
	
	return step;
}

static int
find_star_in_col(const nv_munkres_mask_t *mask, int *row, const int col)
{
	int j;
	*row = -1;

   for (j = 0; j < mask->m; ++j) {
	   if (mask->v[j][col] == NV_MUNKRES_STAR) {
		   *row = j;
	   }
   }
   return *row != -1 ? 1: 0;
}

static int
find_prime_in_row(const nv_munkres_mask_t *mask, const int row, int *col)
{
   int i = 0;
   *col = -1;
   
   for (i = 0; i < mask->n; ++i) {
	   if (mask->v[row][i] == NV_MUNKRES_PRIME) {
		   *col = i;
	   }
   }
   return *col != -1 ? 1: 0;
}

static void
argument_path(nv_munkres_mask_t *mask, int *path[], int count)
{
	int i;
	for (i = 0; i < count; ++i) {
		if (mask->v[path[0][i]][path[1][i]] == NV_MUNKRES_STAR) {
			mask->v[path[0][i]][path[1][i]] = NV_MUNKRES_ZERO;
		} else {
			mask->v[path[0][i]][path[1][i]] = NV_MUNKRES_STAR;
		}
	}
}

static void
erase_primes(nv_munkres_mask_t *mask)
{
	int j, i;

	for (j = 0; j < mask->m; ++j) {
		for (i = 0; i < mask->n; ++i) {
			if (mask->v[j][i] == NV_MUNKRES_PRIME) {
				mask->v[j][i] = NV_MUNKRES_ZERO;
			}
		}
	}
}

static int
step5(nv_matrix_t *mat,
	  nv_munkres_mask_t *mask,
	  int *r_cov, int *c_cov,
	  int **path,
	  int *z_0r, int *z_0c)
{
	int count = 0;
	int j = -1, i = -1;
	int done = 0;
	
	path[0][count] = *z_0r;
	path[1][count] = *z_0c;

	do {
		find_star_in_col(mask, &j, path[1][count]);
		if (j > -1) {
			++count;
			path[0][count] = j;
			path[1][count] = path[1][count - 1];
			
			find_prime_in_row(mask, path[0][count], &i);
			++count;
			path[0][count] = path[0][count - 1];
			path[1][count] = i;
		} else {
			done = 1;
		}
	} while (!done);
	
	argument_path(mask, path, count + 1);
	clear_covers(r_cov, c_cov, NV_MAX(mat->n, mat->m));
	erase_primes(mask);
	
	return 3;/* 3へ */
}

static float
find_smallest(const nv_matrix_t *mat,
			  const int *r_cov,
			  const int *c_cov)
{
	int j, i;
	float minval = FLT_MAX;
	
	for (j = 0; j < mat->m; ++j) {
		if (r_cov[j] == 0) {
			for (i = 0; i < mat->n; ++i) {
				if (c_cov[i] == 0 &&
					NV_MAT_V(mat, j, i) < minval)
				{
					minval = NV_MAT_V(mat, j, i);
				}
			}
		}
	}
	return minval;
}

static int
step6(nv_matrix_t *mat,
	  int *r_cov, int *c_cov)
{
	int j, i;
	float minval = find_smallest(mat, r_cov, c_cov);
	
	for (j = 0; j < mat->m; ++j) {
		for (i = 0; i < mat->n; ++i) {
			if (r_cov[j] == 1) {
				NV_MAT_V(mat, j, i) += minval;
			}
			if (c_cov[i] == 0) {
				NV_MAT_V(mat, j, i) -= minval;
			}
		}
	}
	
	return 4; /* 4へ */
}

float
nv_munkres(nv_matrix_t *task, const nv_matrix_t *cost_matrix)
{
	int mat_n = NV_MAX(cost_matrix->n, cost_matrix->m);
	nv_matrix_t *mat = nv_matrix_dup(cost_matrix);
	nv_munkres_mask_t mask;
	int *path[2];
	int *r_cov = nv_alloc_type(int, mat_n);
	int *c_cov = nv_alloc_type(int, mat_n);
	int step = 1;
	int z_0r = 0;
	int z_0c = 0;
	int j, i;
	float cost;
	nv_matrix_t *min_cost = nv_matrix_alloc(cost_matrix->m, 1);
	int *assign = nv_alloc_type(int, cost_matrix->m);
	
	NV_ASSERT(task->n == cost_matrix->n);
	
	mask.n = mat->n;
	mask.m = mat->m;
	mask.v = nv_alloc_type(int *, mask.m);
	for (j = 0; j < mat->m; ++j) {
		mask.v[j] = nv_alloc_type(int, mask.n);
		memset(mask.v[j], 0, sizeof(int) * mask.n);
	}
	
	path[0] = nv_alloc_type(int, mat_n * mat_n * mat_n);
	path[1] = nv_alloc_type(int, mat_n * mat_n * mat_n);
	memset(path[0], 0, sizeof(int) * mat_n * mat_n * 2 + 1);
	memset(path[1], 0, sizeof(int) * mat_n * mat_n * 2 + 1);
	
	clear_covers(r_cov, c_cov, mat_n);
	
	nv_matrix_zero(task);
	do {
		switch (step) {
        case 1:
			step = step1(mat);
			break;
		case 2:
			step = step2(mat, &mask, r_cov, c_cov);
			break;
		case 3:
			step = step3(&mask, c_cov);
			break;
		case 4:
			step = step4(mat, &mask, r_cov, c_cov, &z_0r, &z_0c);
			break;
		case 5:
			step = step5(mat, &mask, r_cov, c_cov, path, &z_0r, &z_0c);
			break;
		case 6:
			step = step6(mat, r_cov, c_cov);
			break;
		default:
			break;
		}
	} while (step != 0);
	for (j = 0; j < mat->m; ++j) {
		NV_MAT_V(min_cost, 0, j) = FLT_MAX;
		for (i = 0; i < mat->n; ++i) {
			if (mask.v[j][i] == NV_MUNKRES_STAR) {
				NV_MAT_V(task, 0, i) = (float)j;
				if (NV_MAT_V(min_cost, 0, j) > NV_MAT_V(cost_matrix, j, i)) {
					NV_MAT_V(min_cost, 0, j) = NV_MAT_V(cost_matrix, j, i);
				}
			}
		}
	}
	memset(assign, 0, sizeof(int) * cost_matrix->m);
	cost = 0.0f;
	for (i = 0; i < task->n; ++i) {
		j = NV_MAT_VI(task, 0, i);
		if (assign[j] == 0 &&
			!(NV_MAT_V(cost_matrix, j, i) > NV_MAT_V(min_cost, 0, j)))
		{
			cost += NV_MAT_V(cost_matrix, j, i);
			assign[j] = 1;
		} else {
			NV_MAT_V(task, 0, i) = -1;
		}
	}
	for (j = 0; j < mask.m; ++j) {
		nv_free(mask.v[j]);
	}
	nv_free(mask.v);
	
	nv_matrix_free(&min_cost);
	nv_matrix_free(&mat);
	nv_free(r_cov);
	nv_free(c_cov);
	nv_free(path[0]);
	nv_free(path[1]);
	nv_free(assign);
	
	return cost;
}
