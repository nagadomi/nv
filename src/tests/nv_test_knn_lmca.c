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

#define KNN_K 8
#define NK 16
#define MK 32
#define DIM  128
#define MARGIN 1.0f
#define PUSH_RATIO1 0.8f
#define PUSH_RATIO 0.8f
#define DELTA 0.1f
#define DELTA1 0.1f
#define EPOCH 30

void
nv_test_knn_lmca(const nv_matrix_t *train_data,
				 const nv_matrix_t *train_labels,
				 const nv_matrix_t *test_data,
				 const nv_matrix_t *test_labels)
{
	nv_matrix_t *train_data_lmca = nv_matrix_alloc(DIM, train_data->m);
	nv_matrix_t *vec = nv_matrix_alloc(DIM, 1);
	nv_matrix_t *l = nv_matrix_alloc(train_data->n, DIM);
	int i, ok;
	nv_knn_result_t results[KNN_K];
	long t;
	
	NV_TEST_NAME;
	nv_lmca_progress(1);
	
	printf("train: %d, test: %d, %ddim ->  %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n,
		   DIM
		);
	t = nv_clock();
	nv_lmca(l, train_data, train_labels, NK, MK, MARGIN, PUSH_RATIO, DELTA, EPOCH);
	printf("-- %ldms\n", nv_clock() - t);
	nv_lmca_projection_all(train_data_lmca, l, train_data);
	
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
	
	nv_matrix_free(&train_data_lmca);
	nv_matrix_free(&vec);
	nv_matrix_free(&l);
	
	fflush(stdout);
}

void
nv_test_knn_2pass_lmca(const nv_matrix_t *train_data,
					   const nv_matrix_t *train_labels,
					   const nv_matrix_t *test_data,
					   const nv_matrix_t *test_labels)
{
	nv_matrix_t *train_data2 = nv_matrix_clone(train_data);
	nv_matrix_t *test_data2 = nv_matrix_clone(test_data);
	nv_matrix_t *train_data_lmca = nv_matrix_alloc(DIM, train_data->m);
	nv_matrix_t *vec = nv_matrix_alloc(DIM, 1);
	nv_matrix_t *l1 = nv_matrix_alloc(train_data->n, train_data->n);
	nv_matrix_t *l2 = nv_matrix_alloc(train_data->n, DIM);
	int i, ok;
	nv_knn_result_t results[KNN_K];
	long t;
	
	NV_TEST_NAME;
	nv_lmca_progress(1);
	
	printf("train: %d, test: %d, %ddim ->  %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n,
		   DIM
		);
	
	t = nv_clock();
	nv_lmca_init_diag1(l1);
	nv_lmca_train_ex(l1, NV_LMCA_DIAG,
					 train_data, train_labels, NK, MK, MARGIN, PUSH_RATIO1, DELTA1, EPOCH);
	for (i = 0; i < train_data->m; ++i) {
		nv_lmca_projection(train_data2, i, l1, train_data, i);
	}
	for (i = 0; i < test_data->m; ++i) {
		nv_lmca_projection(test_data2, i, l1, test_data, i);
	}
	nv_vector_normalize_all(train_data2);
	nv_vector_normalize_all(test_data2);
	
	nv_lmca(l2, train_data2, train_labels, NK, MK, MARGIN, PUSH_RATIO, DELTA, EPOCH);
	printf("-- %ldms\n", nv_clock() - t);
	for (i = 0; i < train_data2->m; ++i) {
		nv_lmca_projection(train_data_lmca, i, l2, train_data2, i);
#if 0
		printf("%d %f %f\n", NV_MAT_VI(train_labels, i, 0),
			   NV_MAT_V(train_data_lmca, i, 0),
			   NV_MAT_V(train_data_lmca, i, 1));
#endif
	}
	ok = 0;
	for (i = 0; i < test_data2->m; ++i) {
		int knn[NV_TEST_DATA_K] = {0};
		int j, n, max_v, max_i;

		nv_lmca_projection(vec, 0, l2, test_data2, i);
		
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
		   (float)ok / test_data2->m * 100.0f,
		   ok, test_data2->m);

	nv_matrix_free(&train_data2);
	nv_matrix_free(&test_data2);
	nv_matrix_free(&train_data_lmca);
	nv_matrix_free(&vec);
	nv_matrix_free(&l1);
	nv_matrix_free(&l2);
	
	fflush(stdout);
}
