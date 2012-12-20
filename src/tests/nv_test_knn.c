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
#include "nv_num.h"
#include "nv_test.h"

static void
nv_test_knn_nn(const nv_matrix_t *data)
{
	int i;

	NV_TEST_NAME;
	
	for (i = 0; i < data->m; ++i) {
		int nn;
		
		nn = nv_nn(data, data, i);
		NV_ASSERT(nv_vector_eq(data, i, data, nn));
		
		nn = nv_nn_ex(data, data, i, nv_cosine);
		NV_ASSERT(nv_vector_eq(data, i, data, nn));
		
		nn = nv_nn_ex(data, data, i, nv_euclidean);
		NV_ASSERT(nv_vector_eq(data, i, data, nn));
		
		nn = nv_nn_ex(data, data, i, nv_euclidean2);
		NV_ASSERT(nv_vector_eq(data, i, data, nn));
	}
}

static void
nv_test_knn_knn(const nv_matrix_t *data)
{
	int k = 10;
	nv_knn_result_t *results = nv_alloc_type(nv_knn_result_t, k);
	int i;

	NV_TEST_NAME;
	
	for (i = 0; i < data->m; ++i) {
		int n;
		int j;
		
		n = nv_knn(results, k, data, data, i);
		NV_ASSERT(n == k);
		NV_ASSERT(nv_vector_eq(data, i, data, results[0].index));
		for (j = 1; j < k; ++j) {
			NV_ASSERT(results[j - 1].dist <= results[j].dist);
		}
		
		n = nv_knn_ex(results, k, data, data, i, nv_cosine);
		NV_ASSERT(n == k);
		NV_ASSERT(nv_vector_eq(data, i, data, results[0].index));
		for (j = 1; j < k; ++j) {
			NV_ASSERT(results[j - 1].dist <= results[j].dist);
		}
		
		n = nv_knn_ex(results, k, data, data, i, nv_euclidean);
		NV_ASSERT(n == k);
		NV_ASSERT(nv_vector_eq(data, i, data, results[0].index));
		for (j = 1; j < k; ++j) {
			NV_ASSERT(results[j - 1].dist <= results[j].dist);
		}
		
		n = nv_knn_ex(results, k, data, data, i, nv_euclidean2);
		NV_ASSERT(n == k);
		NV_ASSERT(nv_vector_eq(data, i, data, results[0].index));
		for (j = 1; j < k; ++j) {
			NV_ASSERT(results[j - 1].dist <= results[j].dist);
		}
	}
	nv_free(results);
}

void nv_test_knn(const nv_matrix_t *data)
{
	nv_test_knn_nn(data);
	nv_test_knn_knn(data);

	fflush(stdout);
}
