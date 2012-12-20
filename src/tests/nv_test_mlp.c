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

#define HIDDEN_UNIT 128

void
nv_test_mlp(const nv_matrix_t *train_data,
			const nv_matrix_t *train_labels,
			const nv_matrix_t *test_data,
			const nv_matrix_t *test_labels)
{
	nv_mlp_t *mlp = nv_mlp_alloc(train_data->n, HIDDEN_UNIT, NV_TEST_DATA_K);
	nv_matrix_t *ir = nv_matrix_alloc(1, mlp->output == 1 ? 2 : mlp->output);
	nv_matrix_t *hr = nv_matrix_alloc(1, mlp->output == 1 ? 2 : mlp->output);
	int i, ok;
	
	NV_TEST_NAME;
	
	printf("train: %d, test: %d, %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n);

	nv_mlp_progress(1);
	nv_mlp_gaussian_init(mlp, 1.0f, (int)sqrtf(train_data->n), (int)sqrtf(train_data->n), 1);
	nv_matrix_fill(ir, 0.2f);
	nv_matrix_fill(hr, 0.1f);
	nv_mlp_train_ex(mlp, train_data, train_labels, ir, hr, 1, 0, 150, 200);
	nv_matrix_fill(ir, 0.01f);
	nv_matrix_fill(hr, 0.01f);
	nv_mlp_train_ex(mlp, train_data, train_labels, ir, hr, 1, 150, 200, 200);
	
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

	nv_matrix_free(&ir);
	nv_matrix_free(&hr);
	nv_mlp_free(&mlp);
	fflush(stdout);
}
