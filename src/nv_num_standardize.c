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

#include "nv_core.h"
#include "nv_num_vector.h"
#include "nv_num_matrix.h"
#include "nv_num_standardize.h"

void
nv_standardize(nv_matrix_t *x, int x_j,
			   const nv_matrix_t *mean, int mean_j,
			   const nv_matrix_t *sd, int sd_j)
{
	nv_vector_sub(x, x_j, x, x_j, mean, mean_j);
	nv_vector_div(x, x_j, x, x_j, sd, sd_j);
}

void
nv_standardize_local(nv_matrix_t *x, int x_j, float epsilon)
{
	float mean = nv_vector_mean(x, x_j);
	float var = nv_vector_var_ex(x, x_j, mean);
	float sd = sqrtf(var + epsilon);
	nv_vector_subs(x, x_j, x, x_j, mean);
	nv_vector_divs(x, x_j, x, x_j, sd);
}

void
nv_standardize_train(nv_matrix_t *mean, int mean_j,
					 nv_matrix_t *sd, int sd_j,
					 const nv_matrix_t *data,
					 float epsilon)
{
	nv_matrix_mean(mean, mean_j, data);
	nv_matrix_var_ex(sd, sd_j, data, mean, mean_j);
	nv_vector_adds(sd, sd_j, sd, sd_j, epsilon);
	nv_vector_sqrt(sd, sd_j, sd, sd_j);
}

void
nv_standardize_all(nv_matrix_t *a,
				   const nv_matrix_t *mean, int mean_j,
				   const nv_matrix_t *sd, int sd_j)
{
	int j;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (j = 0; j < a->m; ++j) {
		nv_standardize(a, j, mean, mean_j, sd, sd_j);
	}
}

void
nv_standardize_local_all(nv_matrix_t *a, float epsilon)
{
	int j;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (j = 0; j < a->m; ++j) {
		nv_standardize_local(a, j, epsilon);
	}
}


