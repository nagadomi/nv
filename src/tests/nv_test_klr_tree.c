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

static void nv_test_klr_tree_ex(const nv_matrix_t *data, const nv_matrix_t *labels,
								int *width, int height)
{
	nv_matrix_t *cluster_labels = nv_matrix_alloc(1, data->m);
	nv_klr_tree_t *tree = nv_klr_tree_alloc(data->n, width, height);
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

	nv_klr_tree_train(tree, data, NV_LR_PARAM(4, 0.1f, NV_LR_REG_L1, 0.0001f, 1), 100);
	for (i = 0; i < data->m; ++i) {
		NV_MAT_V(cluster_labels, i, 0) = (float)nv_klr_tree_predict_label(tree, data, i);
	}
	purity = nv_purity(K, NV_TEST_DATA_K, cluster_labels, labels);
	printf("purity: %f\n", purity);
	NV_ASSERT(purity > 0.5f);
	
	nv_klr_tree_free(&tree);
	nv_matrix_free(&cluster_labels);
}

static void
nv_test_klr_tree_inherit(const nv_matrix_t *data,const nv_matrix_t *labels,
						 nv_klr_tree_t *tree, const nv_klr_tree_t *base)
{
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

	if (base == NULL) {
		nv_klr_tree_train(tree, data, NV_LR_PARAM(4, 0.1f, NV_LR_REG_L1, 0.0001f, 1), 100);
	} else {
		nv_klr_tree_inherit_train(tree, base,
								  data, NV_LR_PARAM(4, 0.1f, NV_LR_REG_L1, 0.0001f, 1), 100);
	}
	for (i = 0; i < data->m; ++i) {
		NV_MAT_V(cluster_labels, i, 0) = (float)nv_klr_tree_predict_label(tree, data, i);
	}
	purity = nv_purity(K, NV_TEST_DATA_K, cluster_labels, labels);
	printf("purity: %f\n", purity);
	NV_ASSERT(purity > 0.5f);
	
	nv_matrix_free(&cluster_labels);
}

static void
nv_test_klr_tree_nodes(const nv_matrix_t *data, const nv_matrix_t *labels)
{
	int width0[] = { 32 };	
	int width1[] = { 16, 2 };
	int width2[] = { 8, 2, 2 };
	int width3[] = { 4, 4, 2 };
	int width4[] = { 2, 2, 2, 2 };
	int width5[] = { 2, 16 };
	int width6[] = { 2, 2, 8 };
	int width7[] = { 2, 4, 4 };
	
	nv_test_klr_tree_ex(data, labels, width0, 1);
	nv_test_klr_tree_ex(data, labels, width1, 2);
	nv_test_klr_tree_ex(data, labels, width2, 3);
	nv_test_klr_tree_ex(data, labels, width3, 3);
	nv_test_klr_tree_ex(data, labels, width4, 4);
	nv_test_klr_tree_ex(data, labels, width5, 2);
	nv_test_klr_tree_ex(data, labels, width6, 3);
	nv_test_klr_tree_ex(data, labels, width7, 3);
}

static void
nv_test_klr_tree_inherit_all(const nv_matrix_t *data, const nv_matrix_t *labels)
{
	int width0[] = { 8, 4 };
	int width1[] = { 8, 2, 2 };
	int width2[] = { 4, 2, 4 };
	int width3[] = { 4, 2, 2, 2 }; 
	nv_klr_tree_t *tree0 = nv_klr_tree_alloc(data->n, width0, sizeof(width0)/ sizeof(int));
	nv_klr_tree_t *tree1 = nv_klr_tree_alloc(data->n, width1, sizeof(width1)/ sizeof(int));
	nv_klr_tree_t *tree2 = nv_klr_tree_alloc(data->n, width2, sizeof(width2)/ sizeof(int));
	nv_klr_tree_t *tree3 = nv_klr_tree_alloc(data->n, width3, sizeof(width3)/ sizeof(int));
	
	nv_test_klr_tree_inherit(data, labels, tree0, NULL);
	nv_test_klr_tree_inherit(data, labels, tree1, tree0);	
	nv_test_klr_tree_inherit(data, labels, tree2, NULL);
	nv_test_klr_tree_inherit(data, labels, tree3, tree2);

	nv_klr_tree_free(&tree0);
	nv_klr_tree_free(&tree1);
	nv_klr_tree_free(&tree2);
	nv_klr_tree_free(&tree3);
}

void
nv_test_klr_tree(const nv_matrix_t *data, const nv_matrix_t *labels)
{
	nv_test_klr_tree_inherit_all(data, labels);
	nv_test_klr_tree_nodes(data, labels);
	fflush(stdout);
}
