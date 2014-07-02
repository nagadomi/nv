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

#define K 31

void nv_test_klr(const nv_matrix_t *data, const nv_matrix_t *labels)
{
	nv_matrix_t *cluster_labels = nv_matrix_alloc(1, data->m);
	nv_matrix_t *count = nv_matrix_alloc(1, K);
	nv_lr_t *lr = nv_lr_alloc(data->n, K);
	float purity;

	NV_TEST_NAME;

	nv_lr_progress(0);
	nv_klr_train(lr, count, cluster_labels, data,
				 NV_LR_PARAM(2, 0.15f, NV_LR_REG_L2, 0.000001f, 1), 50);
	
	purity = nv_purity(K, NV_TEST_DATA_K, cluster_labels, labels);
	printf("purity: %f\n", purity);
	NV_ASSERT(purity > 0.6f);
	
	nv_lr_free(&lr);
	nv_matrix_free(&cluster_labels);
	nv_matrix_free(&count);

	fflush(stdout);
}
