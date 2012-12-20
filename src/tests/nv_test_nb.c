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

// 

void nv_test_nb(const nv_matrix_t *train_data,
				const nv_matrix_t *train_labels,
				const nv_matrix_t *test_data,
				const nv_matrix_t *test_labels)
{
	nv_nb_t *nb = nv_nb_alloc(train_data->n, NV_TEST_DATA_K);
	int i, ok;
	
	NV_TEST_NAME;
	
	printf("train: %d, test: %d, %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n);
	
	nv_nb_train_all(nb, NV_TEST_DATA_K, train_data, train_labels);
	
	ok = 0;
	for (i = 0; i < test_data->m; ++i) {
		if (nv_nb_predict_label(nb, test_data, i, 0)
			== (int)NV_MAT_V(test_labels, i, 0))
		{
			++ok;
		}
	}
	printf("Accuracy = %f%% (%d/%d)\n",
		   (float)ok / test_data->m * 100.0f,
		   ok, test_data->m);
	
	NV_ASSERT((float)ok / test_data->m > 0.7f);

	nv_nb_free(&nb);
	fflush(stdout);
}
