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

void nv_test_lbgu(const nv_matrix_t *data, const nv_matrix_t *labels)
{
	nv_matrix_t *cluster_labels = nv_matrix_alloc(1, data->m);
	nv_matrix_t *centroids = nv_matrix_alloc(data->n, K);
	nv_matrix_t *count = nv_matrix_alloc(1, K);
	float purity;
	
	NV_TEST_NAME;

	nv_lbgu(centroids, count, cluster_labels, data, K, 50, 50);
	
	purity = nv_purity(K, NV_TEST_DATA_K, cluster_labels, labels);
	printf("purity: %f\n", purity);
	NV_ASSERT(purity > 0.7f);
	
	nv_matrix_free(&cluster_labels);
	nv_matrix_free(&centroids);
	nv_matrix_free(&count);

	fflush(stdout);
}
