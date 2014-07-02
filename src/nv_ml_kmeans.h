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

#ifndef NV_ML_KMEANS
#define NV_ML_KMEANS

#ifdef __cplusplus
extern "C" {
#endif
#include "nv_core.h"
#include "nv_ml.h"

void nv_kmeans_progress(int onoff);

int nv_kmeans(nv_matrix_t *means,  /* k */
			  nv_matrix_t *count,  /* k */
			  nv_matrix_t *labels, /* data->m */
			  const nv_matrix_t *data,
			  const int k,
			  const int max_epoch);

void nv_kmeans_init_pp(nv_matrix_t *means, int k,
					   const nv_matrix_t *data, int tries);
void nv_kmeans_init_rand(nv_matrix_t *means);

void nv_kmeans_init_dist(nv_matrix_t *means, int k,
						 const nv_matrix_t *data);

int nv_kmeans_em(nv_matrix_t *means,  /* k */
				 nv_matrix_t *count,  /* k */
				 nv_matrix_t *labels, /* data->m */
				 const nv_matrix_t *data,
				 const int k,
				 const int max_epoch);

#ifdef __cplusplus
}
#endif


#endif
