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

#undef NDEBUG
#include "nv_core.h"
#include "nv_io.h"
#include "nv_ml.h"
#include "nv_test.h"

void
nv_test_arow(void)
{
	nv_matrix_t *data = nv_load_matrix(NV_TEST_DATA);
	nv_matrix_t *labels = nv_load_matrix(NV_TEST_LABEL);
	nv_matrix_t *train_data = nv_matrix_alloc(data->n, data->m / 4 * 3);
	nv_matrix_t *train_labels = nv_matrix_alloc(labels->n, labels->m / 4 * 3);
	nv_matrix_t *test_data = nv_matrix_alloc(data->n, data->m - train_data->m);
	nv_matrix_t *test_labels = nv_matrix_alloc(labels->n, labels->m - train_labels->m);
	nv_arow_t *arow = nv_arow_alloc(data->n, NV_TEST_DATA_K);
	int i, ok;
	
	NV_TEST_NAME;
	
	nv_vector_normalize_all(data);
	
	printf("train: %d, test: %d, %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n);
	
	nv_dataset(data, labels,
			   train_data, train_labels,
			   test_data, test_labels);
	
	//nv_arow_progress(1);
	nv_arow_init(arow);
	nv_arow_train(arow,
				  train_data, train_labels, 0.1f, 16);
	ok = 0;
	for (i = 0; i < test_data->m; ++i) {
		if (nv_arow_predict_label(arow, test_data, i) == NV_MAT_VI(test_labels, i, 0)) {
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

	nv_arow_free(&arow);

	fflush(stdout);
}
