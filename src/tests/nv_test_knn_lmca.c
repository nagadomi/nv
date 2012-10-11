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
#include "nv_ml.h"
#include "nv_test.h"

#define KNN_K 3
#define NK 16
#define MK 512
#define DIM  32
#define MARGIN powf(0.3f, 2.0f)
#define PUSH_RATIO 0.5f
#define DELTA 0.5f
#define EPOCH 6

void
nv_test_knn_lmca(void)
{
	nv_matrix_t *data = nv_load_matrix(NV_TEST_DATA);
	nv_matrix_t *labels = nv_load_matrix(NV_TEST_LABEL);
	nv_matrix_t *train_data = nv_matrix_alloc(data->n, data->m / 4 * 3);
	nv_matrix_t *train_labels = nv_matrix_alloc(labels->n, labels->m / 4 * 3);
	nv_matrix_t *test_data = nv_matrix_alloc(data->n, data->m - train_data->m);
	nv_matrix_t *test_labels = nv_matrix_alloc(labels->n, labels->m - train_labels->m);
	nv_matrix_t *train_data_lmca = nv_matrix_alloc(DIM, train_data->m);
	nv_matrix_t *vec = nv_matrix_alloc(DIM, 1);
	nv_matrix_t *l = nv_matrix_alloc(data->n, DIM);
	int i, ok;
	nv_knn_result_t results[KNN_K];
	long t;
	
	NV_TEST_NAME;
	//nv_lmca_progress(1);
	
	nv_vector_normalize_all(data);
	printf("train: %d, test: %d, %ddim ->  %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n,
		   DIM
		);
	nv_dataset(data, labels,
			   train_data, train_labels,
			   test_data, test_labels);
	//nv_matrix_m(train_data, 250);
	//nv_matrix_m(train_labels, 250);
	
	t = nv_clock();
	nv_lmca(l, train_data, train_labels, NK, MK, MARGIN, PUSH_RATIO, DELTA, EPOCH);
	printf("-- %ldms\n", nv_clock() - t);
	for (i = 0; i < train_data->m; ++i) {
		nv_lmca_projection(train_data_lmca, i, l, train_data, i);
#if 0
		printf("%d %f %f\n", NV_MAT_VI(train_labels, i, 0),
			   NV_MAT_V(train_data_lmca, i, 0),
			   NV_MAT_V(train_data_lmca, i, 1));
#endif
	}
	ok = 0;
	for (i = 0; i < test_data->m; ++i) {
		int knn[NV_TEST_DATA_K] = {0};
		int j, n, max_v, max_i;

		nv_lmca_projection(vec, 0, l, test_data, i);
		
		n = nv_knn(results, KNN_K, train_data_lmca, vec, 0);
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
	
	nv_matrix_free(&data);
	nv_matrix_free(&labels);
	nv_matrix_free(&train_data);
	nv_matrix_free(&train_labels);
	nv_matrix_free(&test_data);
	nv_matrix_free(&test_labels);
	nv_matrix_free(&train_data_lmca);
	nv_matrix_free(&vec);
	nv_matrix_free(&l);
	
	fflush(stdout);
}
