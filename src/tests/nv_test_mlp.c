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

#define HIDDEN_UNIT 320

void
nv_test_mlp(const nv_matrix_t *train_data,
			const nv_matrix_t *train_labels,
			const nv_matrix_t *test_data,
			const nv_matrix_t *test_labels)
{
	nv_mlp_t *mlp = nv_mlp_alloc(train_data->n, HIDDEN_UNIT, NV_TEST_DATA_K);
	int i, ok;
	
	NV_TEST_NAME;
	
	printf("train: %d, test: %d, %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n);
	
	nv_mlp_progress(1);
	nv_mlp_init(mlp, train_data);
	nv_mlp_dropout(mlp, 0.5f);
	nv_mlp_noise(mlp, 0.1f);
	nv_mlp_train_ex(mlp, train_data, train_labels, 1.0f, 0.1f, 0, 200, 200);
	//nv_mlp_train_ex(mlp, train_data, train_labels, 2.0f, 0.1f, 20, 180, 200);
	//nv_mlp_train_ex(mlp, train_data, train_labels, 0.001f, 0.001f, 180, 200, 200);

	ok = 0;
	for (i = 0; i < test_data->m; ++i) {
		if (nv_mlp_predict_label(mlp, test_data, i) == (int)NV_MAT_V(test_labels, i, 0)) {
			++ok;
		}
	}
	printf("Accuracy = %f%% (%d/%d)\n",
		   (float)ok / test_data->m * 100.0f,
		   ok, test_data->m);

	NV_ASSERT((float)ok / test_data->m > 0.7f);

	nv_mlp_free(&mlp);
	fflush(stdout);
}
