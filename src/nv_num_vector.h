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

#ifndef NV_NUM_VECTOR_H
#define NV_NUM_VECTOR_H

#include "nv_core.h"
#include "nv_num_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif
	
void 
nv_vector_sqrt(nv_matrix_t *vec0, int m0,
			   const nv_matrix_t *vec1, int m1);
void 
nv_vector_pows(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  float x);
void 
nv_vector_not10(nv_matrix_t *vec0, int m0,
				const nv_matrix_t *vec1, int m1);
	
void
nv_vector_in_range10(nv_matrix_t *dst,
					 int dj,
					 const nv_matrix_t *lower,
					 int lj,
					 const nv_matrix_t *upper,
					 int uj,
					 const nv_matrix_t *vec,
					 int vj);
	
void 
nv_vector_subs(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			   float v);
void 
nv_vector_subsm(nv_matrix_t *vec0, int m0,
				const nv_matrix_t *vec1, int m1,
				float v,
				const nv_matrix_t *mask, int m2
	);
	
void nv_vector_sub(nv_matrix_t *vec0, int m0,
				   const nv_matrix_t *vec1, int m1,
				   const nv_matrix_t *vec2, int m2);
void 
nv_vector_adds(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			   float v);
	
void 
nv_vector_add(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  const nv_matrix_t *vec2, int m2);
	
void 
nv_vector_mul(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  const nv_matrix_t *vec2, int m2);

void 
nv_vector_mulmtr(nv_matrix_t *vec0, int m0,
				 const nv_matrix_t *vec1, int m1,
				 const nv_matrix_t *mat);

void 
nv_vector_div(nv_matrix_t *vec0, int m0,
			  const nv_matrix_t *vec1, int m1,
			  const nv_matrix_t *vec2, int m2);
	
	
float nv_vector_dot(const nv_matrix_t *vec1, int m1, const nv_matrix_t *vec2, int m2);
float nv_vector_norm(const nv_matrix_t *v, int m);

float nv_vector_maxs(const nv_matrix_t *v, int m);
void nv_vector_max(nv_matrix_t *vec0, int m0,
				   const nv_matrix_t *vec1, int m1,
				   const nv_matrix_t *vec2, int m2);
int nv_vector_max_n(const nv_matrix_t *v, int m);
float nv_vector_mins(const nv_matrix_t *v, int m);
void nv_vector_min(nv_matrix_t *vec0, int m0,
				   const nv_matrix_t *vec1, int m1,
				   const nv_matrix_t *vec2, int m2);
int nv_vector_max_n(const nv_matrix_t *v, int m);
int nv_vector_min_n(const nv_matrix_t *v, int m);
int nv_vector_max_m(const nv_matrix_t *v);
int nv_vector_min_m(const nv_matrix_t *v);
void nv_vector_avg(nv_matrix_t *avg, int avg_m, const nv_matrix_t *mat);

int nv_vector_maxnorm_m(const nv_matrix_t *v);
int nv_vector_maxsum_m(const nv_matrix_t *v);

float nv_vector_sum(const nv_matrix_t *x, int j);
float nv_vector_mean(const nv_matrix_t *x, int j);
float nv_vector_var(const nv_matrix_t *x, int j);
float nv_vector_var_ex(const nv_matrix_t *x, int j, float mean);

void nv_vector_normalize_L1(nv_matrix_t *v, int vm);
void nv_vector_normalize_L2(nv_matrix_t *v, int vm);
#define nv_vector_normalize(v, vm) nv_vector_normalize_L2((v), (vm))

void nv_vector_normalize_shift(nv_matrix_t *v, int vm, float min_v, float max_v);
void nv_vector_muls(nv_matrix_t *a, int am, const nv_matrix_t *x, int xm, float v);
void nv_vector_divs(nv_matrix_t *a, int am, const nv_matrix_t *x, int xm, float v);	
void nv_vector_inv(nv_matrix_t *a, int am, const nv_matrix_t *x, int xm);
int64_t nv_float_nonzero_index(const float *v, int64_t s, int64_t e);
int64_t nv_float_find_index(const float *v, int64_t s, int64_t e, float key);
	
int nv_vector_eq(const nv_matrix_t *vec1, int j1, const nv_matrix_t *vec2, int j2);	
nv_int_float_t nv_vector_max_ex(const nv_matrix_t *v, int m);
nv_int_float_t nv_vector_min_ex(const nv_matrix_t *v, int m);

void nv_vector_normalize_all_L1(nv_matrix_t *mat);
void nv_vector_normalize_all_L2(nv_matrix_t *mat);

#define nv_vector_normalize_all(mat) nv_vector_normalize_all_L2(mat)
#define nv_vector_absmax(v, m) NV_MAX(fabsf(nv_vector_max(v, m)), fabsf(nv_vector_min(v, m)))

void nv_vector_clip(nv_matrix_t *v, int v_j, float vmin, float vmax);

#ifdef __cplusplus
}
#endif
#endif
