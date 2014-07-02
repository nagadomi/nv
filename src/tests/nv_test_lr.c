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
nv_test_lr(const nv_matrix_t *train_data,
		   const nv_matrix_t *train_labels,
		   const nv_matrix_t *test_data,
		   const nv_matrix_t *test_labels)
{
	nv_lr_t *lr = nv_lr_alloc(train_data->n, NV_TEST_DATA_K);
	int i, ok;
	
	NV_TEST_NAME;
	
	printf("train: %d, test: %d, %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n);
	
	nv_lr_progress(1);
	nv_lr_init(lr, train_data);
	nv_lr_train(lr,
				train_data, train_labels,
				NV_LR_PARAM(100, 0.1f, NV_LR_REG_L2, 0.0001f, 0));
	ok = 0;
	for (i = 0; i < test_data->m; ++i) {
		if (nv_lr_predict_label(lr, test_data, i) == (int)NV_MAT_V(test_labels, i, 0)) {
			++ok;
		}
	}
	printf("Accuracy = %f%% (%d/%d)\n",
		   (float)ok / test_data->m * 100.0f,
		   ok, test_data->m);

	nv_lr_free(&lr);
	fflush(stdout);
}
