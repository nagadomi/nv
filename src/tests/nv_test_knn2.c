/*
 * This file is part of libnv.
 *
 * Copyright (C) 2012 nagadomi@nurs.or.jp
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

#undef NDEBUG
#include "nv_core.h"
#include "nv_io.h"
#include "nv_num.h"
#include "nv_test.h"

#define KNN_K 5

void
nv_test_knn2(const nv_matrix_t *train_data,
			 const nv_matrix_t *train_labels,
			 const nv_matrix_t *test_data,
			 const nv_matrix_t *test_labels)
{
	nv_knn_result_t results[KNN_K];
	int i, ok;
	
	NV_TEST_NAME;
	
	printf("train: %d, test: %d, %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n);
	
	ok = 0;
	for (i = 0; i < test_data->m; ++i) {
		int knn[NV_TEST_DATA_K] = {0};
		int j, n, max_v, max_i;
		n = nv_knn(results, KNN_K, train_data, test_data, i);
		for (j = 0; j < n; ++j) {
			++knn[NV_MAT_VI(train_labels, results[j].index, 0)];
		}
		max_v = max_i= 0;
		for (j = 0; j < NV_TEST_DATA_K; ++j) {
			if (max_v < knn[j]) {
				max_v = knn[j];
				max_i = j;
			}
		}
		if (max_i == NV_MAT_VI(test_labels, i, 0)) {
			++ok;
		}
	}
	printf("Accuracy = %f%% (%d/%d)\n",
		   (float)ok / test_data->m * 100.0f,
		   ok, test_data->m);

	fflush(stdout);
}
