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

#define NPCA 128
#define K 31

static void
nv_test_pca_kmeans_tree_ex(int *width, int height)
{
	nv_matrix_t *data = nv_load_matrix(NV_TEST_DATA);
	nv_matrix_t *labels = nv_load_matrix(NV_TEST_LABEL);
	nv_matrix_t *cluster_labels = nv_matrix_alloc(1, data->m);
	nv_pca_kmeans_tree_t *tree = nv_pca_kmeans_tree_alloc(data->n, NPCA, width, height);
	float purity;
	int i;

	NV_TEST_NAME;
	printf("tree: ");
	for (i = 0; i < height; ++i) {
		if (i != 0) {
			printf(", ");
		}
		printf("%d", width[i]);
	}
	printf("\n");

	nv_vector_normalize_all(data);

	nv_pca_kmeans_tree_train(tree, data, 100);
	for (i = 0; i < data->m; ++i) {
		NV_MAT_V(cluster_labels, i, 0) = (float)nv_pca_kmeans_tree_predict_label(tree, data, i);
	}
	
	purity = nv_purity(K, NV_TEST_DATA_K, cluster_labels, labels);
	printf("purity: %f\n", purity);
	NV_ASSERT(purity > 0.2f);
	
	nv_pca_kmeans_tree_free(&tree);
	nv_matrix_free(&data);
	nv_matrix_free(&labels);
	nv_matrix_free(&cluster_labels);
}

static void
nv_test_pca_kmeans_tree_inherit(nv_pca_kmeans_tree_t *tree, const nv_pca_kmeans_tree_t *base)
{
	nv_matrix_t *data = nv_load_matrix(NV_TEST_DATA);
	nv_matrix_t *labels = nv_load_matrix(NV_TEST_LABEL);
	nv_matrix_t *cluster_labels = nv_matrix_alloc(1, data->m);
	float purity;
	int i;

	NV_TEST_NAME;
	printf("tree: ");
	for (i = 0; i < tree->height; ++i) {
		if (i != 0) {
			printf(", ");
		}
		printf("%d", tree->dim[i]);
	}
	printf("\n");

	nv_vector_normalize_all(data);

	if (base == NULL) {
		nv_pca_kmeans_tree_train(tree, data, 100);
	} else {
		nv_pca_kmeans_tree_inherit_train(tree, base,
										 data, 100);
	}
	for (i = 0; i < data->m; ++i) {
		NV_MAT_V(cluster_labels, i, 0) = (float)nv_pca_kmeans_tree_predict_label(tree, data, i);
	}
	purity = nv_purity(K, NV_TEST_DATA_K, cluster_labels, labels);
	printf("purity: %f\n", purity);
	NV_ASSERT(purity > 0.2f);
	
	nv_matrix_free(&data);
	nv_matrix_free(&labels);
	nv_matrix_free(&cluster_labels);
}

static void
nv_test_pca_kmeans_tree_nodes(void)
{
	int width0[] = { 32 };	
	int width1[] = { 16, 2 };
	int width2[] = { 8, 2, 2 };
	int width3[] = { 4, 4, 2 };
	int width4[] = { 2, 2, 2, 2 };
	int width5[] = { 2, 16 };
	int width6[] = { 2, 2, 8 };
	int width7[] = { 2, 4, 4 };	
	
	nv_test_pca_kmeans_tree_ex(width0, 1);
	nv_test_pca_kmeans_tree_ex(width1, 2);
	nv_test_pca_kmeans_tree_ex(width2, 3);
	nv_test_pca_kmeans_tree_ex(width3, 3);
	nv_test_pca_kmeans_tree_ex(width4, 4);
	nv_test_pca_kmeans_tree_ex(width5, 2);
	nv_test_pca_kmeans_tree_ex(width6, 3);
	nv_test_pca_kmeans_tree_ex(width7, 3);
}

static void
nv_test_pca_kmeans_tree_inherit_all(void)
{
	nv_matrix_t *data = nv_load_matrix(NV_TEST_DATA);
	int width0[] = { 8, 4 };
	int width1[] = { 8, 2, 2 };
	int width2[] = { 4, 2, 4 };
	int width3[] = { 4, 2, 2, 2 }; 
	nv_pca_kmeans_tree_t *tree0 = nv_pca_kmeans_tree_alloc(data->n, NPCA, width0,
														   sizeof(width0)/ sizeof(int));
	nv_pca_kmeans_tree_t *tree1 = nv_pca_kmeans_tree_alloc(data->n, NPCA, width1,
														   sizeof(width1)/ sizeof(int));
	nv_pca_kmeans_tree_t *tree2 = nv_pca_kmeans_tree_alloc(data->n, NPCA, width2,
														   sizeof(width2)/ sizeof(int));
	nv_pca_kmeans_tree_t *tree3 = nv_pca_kmeans_tree_alloc(data->n, NPCA, width3,
														   sizeof(width3)/ sizeof(int));
	
	nv_test_pca_kmeans_tree_inherit(tree0, NULL);
	nv_test_pca_kmeans_tree_inherit(tree1, tree0);	
	nv_test_pca_kmeans_tree_inherit(tree2, NULL);
	nv_test_pca_kmeans_tree_inherit(tree3, tree2);

	nv_pca_kmeans_tree_free(&tree0);
	nv_pca_kmeans_tree_free(&tree1);
	nv_pca_kmeans_tree_free(&tree3);
	nv_pca_kmeans_tree_free(&tree3);
	nv_matrix_free(&data);
}

void
nv_test_pca_kmeans_tree(void)
{
	nv_test_pca_kmeans_tree_inherit_all();
	nv_test_pca_kmeans_tree_nodes();

	fflush(stdout);
}
