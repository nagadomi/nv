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

#undef NDEBUG
#include "nv_core.h"
#include "nv_io.h"
#include "nv_ml.h"
#include "nv_test.h"

#define MLP_HIDDEN_UNIT 320
#define DAE_HIDDEN_UNIT 512

static void
normalize(nv_matrix_t *data)
{
	int j;
	nv_matrix_t *mean = nv_matrix_alloc(data->n, 1);
	nv_matrix_mean(mean, 0, data);
	for (j = 0; j < data->m; ++j) {
		nv_vector_sub(data, j, data, j, mean, 0);
	}
	nv_matrix_normalize_shift(data, 0.0f, 1.0f);
	nv_matrix_free(&mean);
}

void
nv_test_dae(const nv_matrix_t *train_data,
			const nv_matrix_t *train_labels,
			const nv_matrix_t *test_data,
			const nv_matrix_t *test_labels)
{
	nv_dae_t *dae = nv_dae_alloc(train_data->n, DAE_HIDDEN_UNIT);
	nv_mlp_t *mlp = nv_mlp_alloc(DAE_HIDDEN_UNIT, MLP_HIDDEN_UNIT, NV_TEST_DATA_K);
	nv_matrix_t *scale_train_data = nv_matrix_dup(train_data);
	nv_matrix_t *scale_test_data = nv_matrix_dup(test_data);
	nv_matrix_t *dae_train_data = nv_matrix_alloc(DAE_HIDDEN_UNIT, train_data->m);
	nv_matrix_t *dae_test_data = nv_matrix_alloc(DAE_HIDDEN_UNIT, test_data->m);
	int i, ok;
	
	NV_TEST_NAME;
	
	printf("train: %d, test: %d, %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n);
	normalize(scale_train_data);
	normalize(scale_test_data);
	
	nv_dae_progress(1);
	nv_dae_init(dae, scale_train_data);
	nv_dae_noise(dae, 0.5f);
	nv_dae_sparsity(dae, 0.05f);
	nv_dae_train(dae, scale_train_data, 0.01f, 0, 50, 50);

	for (i = 0; i < train_data->m; ++i) {
		nv_dae_encode(dae, dae_train_data, i, scale_train_data, i);
	}
	for (i = 0; i < test_data->m; ++i) {
		nv_dae_encode(dae, dae_test_data, i, scale_test_data, i);
	}
	nv_mlp_progress(1);
	nv_mlp_init(mlp, dae_train_data);
	nv_mlp_dropout(mlp, 0.5f);
	nv_mlp_noise(mlp, 0.1f);
	nv_mlp_train_ex(mlp, dae_train_data, train_labels, 0.1f, 0.1f, 0, 250, 250);
	ok = 0;
	for (i = 0; i < dae_test_data->m; ++i) {
		if (nv_mlp_predict_label(mlp, dae_test_data, i) == (int)NV_MAT_V(test_labels, i, 0)) {
			++ok;
		}
	}
	printf("Accuracy = %f%% (%d/%d)\n",
		   (float)ok / test_data->m * 100.0f,
		   ok, test_data->m);

	NV_ASSERT((float)ok / test_data->m > 0.7f);

	nv_mlp_free(&mlp);
	nv_dae_free(&dae);
	nv_matrix_free(&dae_test_data);
	nv_matrix_free(&dae_train_data);
	nv_matrix_free(&scale_test_data);
	nv_matrix_free(&scale_train_data);

	fflush(stdout);
}
