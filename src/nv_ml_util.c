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
#include "nv_ml_util.h"

void
nv_dataset(nv_matrix_t *data,
		   nv_matrix_t *labels,
		   nv_matrix_t *train_data,
		   nv_matrix_t *train_labels,
		   nv_matrix_t *test_data,
		   nv_matrix_t *test_labels)
{
	int i, j;
	
	NV_ASSERT(train_data->m + test_data->m <= data->m);
	NV_ASSERT(train_data->m == train_labels->m);
	NV_ASSERT(test_data->m == test_labels->m);
	
	nv_vector_shuffle_pair(data, labels);

	for (i = 0; i < train_data->m; ++i) {
		nv_vector_copy(train_data, i, data, i);
		nv_vector_copy(train_labels, i, labels, i);
	}
	for (j = 0; j < test_data->m; ++j) {
		nv_vector_copy(test_data, j, data, i);
		nv_vector_copy(test_labels, j, labels, i);
		++i;
	}
}

float
nv_purity(int cluster_k,
		  int correct_k,
		  const nv_matrix_t *cluster_labels,
		  const nv_matrix_t *correct_labels)
{
	int i, j;
	int *nir = nv_alloc_type(int, correct_k);
	float purity = 0.0f;
	
	NV_ASSERT(cluster_labels->m == correct_labels->m);
	
	for (i = 0; i < cluster_k; ++i) {
		int max_v = -1;
		int nr = 0;
		
		memset(nir, 0, sizeof(int) * correct_k);
		
		for (j = 0; j < cluster_labels->m; ++j) {
			if ((int)NV_MAT_V(cluster_labels, j, 0) == i) {
				++nir[(int)NV_MAT_V(correct_labels, j, 0)];
				++nr;
			}
		}
		for (j = 0; j < correct_k; ++j) {
			if (max_v < nir[j]) {
				max_v = nir[j];
			}
		}
		if (nr > 0) {
			purity += ((float)nr / cluster_labels->m) * ((float)max_v / nr);
		}
	}
	nv_free(nir);
	
	return purity;
}

