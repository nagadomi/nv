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

#include "nv_core.h"
#include "nv_num.h"
#include "nv_io.h"
#include "nv_test.h"

int main(void)
{
	nv_matrix_t *data = nv_load_matrix(NV_TEST_DATA);
	nv_matrix_t *labels = nv_load_matrix(NV_TEST_LABEL);
	nv_matrix_t *train_data = nv_matrix_alloc(data->n, data->m / 4 * 3);
	nv_matrix_t *train_labels = nv_matrix_alloc(labels->n, labels->m / 4 * 3);
	nv_matrix_t *test_data = nv_matrix_alloc(data->n, data->m - train_data->m);
	nv_matrix_t *test_labels = nv_matrix_alloc(labels->n, labels->m - train_labels->m);

	nv_vector_normalize_all(data);
	nv_srand(11);
	nv_dataset(data, labels,
			   train_data, train_labels,
			   test_data, test_labels);
	NV_BACKTRACE;
	nv_test_sha1();
	nv_test_io();
	nv_test_serialize();
	nv_test_eigen();
	nv_test_matrix();
	nv_test_keypoint();
	
	nv_test_knn_pca(train_data, train_labels, test_data, test_labels);
	nv_test_knn2(train_data, train_labels, test_data, test_labels);
	
#ifdef NV_TEST_ML
	nv_test_knn_lmca(train_data, train_labels, test_data, test_labels);
	nv_test_knn_2pass_lmca(train_data, train_labels, test_data, test_labels);
	
	nv_test_lr(train_data, train_labels, test_data, test_labels);
	nv_test_arow(train_data, train_labels, test_data, test_labels);
	nv_test_mlp(train_data, train_labels, test_data, test_labels);
	nv_test_dae(train_data, train_labels, test_data, test_labels);
	nv_test_nb(train_data, train_labels, test_data, test_labels);
	
	nv_test_knn(train_data);
	nv_test_kmeans(train_data, train_labels, NV_TEST_DATA_K);
	nv_test_lbgu(train_data, train_labels);	
	nv_test_klr(train_data, train_labels);
	nv_test_knb(train_data, train_labels);

	nv_test_kmeans_tree(train_data, train_labels);
	nv_test_pca_kmeans_tree(train_data, train_labels);
	nv_test_klr_tree(train_data, train_labels);
	
	//nv_test_plsi();
#endif	
	nv_test_munkres();
	
	nv_matrix_free(&data);
	nv_matrix_free(&labels);
	nv_matrix_free(&train_data);
	nv_matrix_free(&train_labels);
	nv_matrix_free(&test_data);
	nv_matrix_free(&test_labels);
	
	return 0;
}
