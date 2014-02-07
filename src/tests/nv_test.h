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

#ifndef NV_TEST_H
#define NV_TEST_H

#define NV_TEST_NAME (fprintf(stdout, "---- %s\n", __FUNCTION__))

#define NV_TEST_EQ0(a) (!(fabsf(v) > 0.000002f))
#define NV_TEST_EQ(a, b) (fabsf(a - b) <= 16.0f * FLT_EPSILON * NV_MAX(fabsf(a), fabsf(b)))

void nv_test_sha1(void);
void nv_test_serialize(void);
void nv_test_lr(const nv_matrix_t *train_data,
				const nv_matrix_t *train_labels,
				const nv_matrix_t *test_data,
				const nv_matrix_t *test_labels);
void nv_test_arow(const nv_matrix_t *train_data,
				  const nv_matrix_t *train_labels,
				  const nv_matrix_t *test_data,
				  const nv_matrix_t *test_labels);

void nv_test_mlp(const nv_matrix_t *train_data,
				 const nv_matrix_t *train_labels,
				 const nv_matrix_t *test_data,
				 const nv_matrix_t *test_labels);
void nv_test_dae(const nv_matrix_t *train_data,
				 const nv_matrix_t *train_labels,
				 const nv_matrix_t *test_data,
				 const nv_matrix_t *test_labels);

void nv_test_nb(const nv_matrix_t *train_data,
				const nv_matrix_t *train_labels,
				const nv_matrix_t *test_data,
				const nv_matrix_t *test_labels);
void nv_test_keypoint(void);
void nv_test_knn(const nv_matrix_t *data);
void nv_test_kmeans(const nv_matrix_t *data, const nv_matrix_t *labels, int k);
void nv_test_klr(const nv_matrix_t *data, const nv_matrix_t *labels);
void nv_test_klr_tree(const nv_matrix_t *data, const nv_matrix_t *labels);
void nv_test_kmeans_tree(const nv_matrix_t *data, const nv_matrix_t *labels);
void nv_test_pca_kmeans_tree(const nv_matrix_t *data, const nv_matrix_t *labels);
void nv_test_lbgu(const nv_matrix_t *data, const nv_matrix_t *labels);
void nv_test_knb(const nv_matrix_t *data, const nv_matrix_t *labels);
void nv_test_plsi(const nv_matrix_t *data, const nv_matrix_t *labels);
void nv_test_io(void);
void nv_test_munkres(void);
void nv_test_knn2(const nv_matrix_t *train_data,
				  const nv_matrix_t *train_labels,
				  const nv_matrix_t *test_data,
				  const nv_matrix_t *test_labels);
void nv_test_knn_pca(const nv_matrix_t *train_data,
					 const nv_matrix_t *train_labels,
					 const nv_matrix_t *test_data,
					 const nv_matrix_t *test_labels);

void nv_test_knn_lmca(const nv_matrix_t *train_data,
					  const nv_matrix_t *train_labels,
					  const nv_matrix_t *test_data,
					  const nv_matrix_t *test_labels);
void nv_test_knn_2pass_lmca(const nv_matrix_t *train_data,
							const nv_matrix_t *train_labels,
							const nv_matrix_t *test_data,
							const nv_matrix_t *test_labels);
void nv_test_eigen(void);
void nv_test_matrix(void);

#endif
