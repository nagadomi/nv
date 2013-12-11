/*
 * This file is part of libnv.
 *
 * Copyright (C) 2011 nagadomi@nurs.or.jp
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
#include "nv_ml.h"

#define NV_PCA_KMEANS_TREE_TOPN 1

static int nv_pca_kmeans_tree_progress_flag = 0;

void
nv_pca_kmeans_tree_progress(int flag)
{
	nv_pca_kmeans_tree_progress_flag = flag;
}

static void
pca_projection(const nv_matrix_t *eigen_vec,
			   nv_matrix_t *y, int yj,
			   const nv_matrix_t *x, int xj)
{
	int i;
	
	NV_ASSERT(eigen_vec->m == y->n);
	NV_ASSERT(eigen_vec->n == x->n);
	
	for (i = 0; i < eigen_vec->m; ++i) {
		NV_MAT_V(y, yj, i) = nv_vector_dot(eigen_vec, i, x, xj);
	}
}

static void
pca_projection_all(const nv_matrix_t *eigen_vec,
				   nv_matrix_t *data_pca,
				   const nv_matrix_t *data)
{
	int j;
	
	NV_ASSERT(eigen_vec->m == data_pca->n);
	NV_ASSERT(eigen_vec->n == data->n);
	NV_ASSERT(data_pca->m == data->m);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (j = 0; j < data->m; ++j) {
		pca_projection(eigen_vec, data_pca, j, data, j);
	}
}

nv_pca_kmeans_tree_t *
nv_pca_kmeans_tree_alloc(int n, int d, const int *dim, int height)
{
	nv_pca_kmeans_tree_t *tree = nv_alloc_type(nv_pca_kmeans_tree_t, 1);
	int y, x;
	int m = 1;
	int prev_width = 1;
	
	tree->height = height;
	tree->dim = nv_alloc_type(int, height);
	tree->node = nv_alloc_type(int, height);
	tree->mat = nv_alloc_type(nv_matrix_t **, tree->height);
	tree->eigen_vec = nv_alloc_type(nv_matrix_t **, tree->height);
	
	for (y = 0; y < height; ++y) {
		tree->dim[y] = dim[y];
		tree->node[y] = prev_width;
		tree->mat[y] = nv_alloc_type(nv_matrix_t *, prev_width);
		tree->eigen_vec[y] = nv_alloc_type(nv_matrix_t *, prev_width);
		
		for (x = 0; x < prev_width; ++x) {
			tree->mat[y][x] = nv_matrix_alloc(d, dim[y]);
			tree->eigen_vec[y][x] = nv_matrix_alloc(n, d);
		}
		
		prev_width = dim[y] * prev_width;
		m *= dim[y];
	}
	tree->d = d;
	tree->n = n;
	tree->d = d;
	tree->m = m;
	if (nv_pca_kmeans_tree_progress_flag) {
		printf("nv_pca_kmeans_tree_alloc: alloc: %d dim\n", tree->m);
		fflush(stdout);
	}
	
	return tree;
}

int
nv_pca_kmeans_tree_predict_label_ex(const nv_pca_kmeans_tree_t *tree, int height,
									const nv_matrix_t *vec, int vec_j, int nn)
{
	int y, x;
	int idx = 0;
	nv_matrix_t *vec_pca = nv_matrix_alloc(tree->d, 1);
	
	// 0 ,1
	NV_ASSERT(height != 0);
	for (y = 0; y < height; ++y) {
		pca_projection(tree->eigen_vec[y][idx], vec_pca, 0, vec, vec_j);
		//pca_projection(tree->eigen_vec[0][0], vec_pca, 0, vec, vec_j);
		if (y + 2 == height) {
			if (nn <= 1) {
				x = nv_nn(tree->mat[y][idx], vec_pca, 0);
				idx *= tree->dim[y];
				idx += x;

				pca_projection(tree->eigen_vec[y + 1][idx], vec_pca, 0, vec, vec_j);
				x = nv_nn(tree->mat[y + 1][idx], vec_pca, 0);
				idx *= tree->dim[y + 1];
				idx += x;
			} else {
				nv_knn_result_t *results = nv_alloc_type(nv_knn_result_t, nn);
				int i;
				int min_index = -1;
				float min_dist = FLT_MAX;
				int min_x = 0;
				
				nn = nv_knn(results, nn,
							tree->mat[y][idx], vec_pca, 0);
				idx *= tree->dim[y];
				
				
				for (i = 0; i < nn; ++i) {
					int tmp_idx = idx + results[i].index;
					nv_int_float_t ld;
					
					pca_projection(tree->eigen_vec[y + 1][idx], vec_pca, 0, vec, vec_j);
					ld = nv_nn_dist(tree->mat[y + 1][tmp_idx],
									vec_pca, 0);
					if (ld.f < min_dist) {
						min_x = results[i].index;
						min_dist = ld.f;
						min_index = ld.i;
					}
				}
				idx += min_x;
				idx *= tree->dim[y + 1];
				idx += min_index;
				nv_free(results);
			}
			break;
		} else {
			x = nv_nn(tree->mat[y][idx], vec_pca, 0);
			idx *= tree->dim[y];
			idx += x;
		}
	}
	nv_matrix_free(&vec_pca);
	
	return idx;
}

nv_int_float_t
nv_pca_kmeans_tree_predict_label_and_dist_ex(const nv_pca_kmeans_tree_t *tree, int height,
											 const nv_matrix_t *vec, int vec_j, int nn)
{
	int y, x, idx = 0;
	nv_int_float_t ld;
	
	ld.i = 0;
	ld.f = 0.0f;
	
	// 0 ,1
	for (y = 0; y < height; ++y) {
		if (y + 2 == height) {
			if (nn <= 1) {
				x = nv_nn(tree->mat[y][idx], vec, vec_j);
				idx *= tree->dim[y];
				idx += x;
				
				ld = nv_nn_dist(tree->mat[y + 1][idx], vec, vec_j);
				x = ld.i;
				idx *= tree->dim[y + 1];
				idx += x;
				ld.i = idx;
			} else {
				nv_knn_result_t *results = nv_alloc_type(nv_knn_result_t, nn);
				int i;
				int min_index = -1;
				float min_dist = FLT_MAX;
				int min_x = 0;
				
				nn = nv_knn(results, nn,
							tree->mat[y][idx], vec, vec_j);
				idx *= tree->dim[y];
				for (i = 0; i < nn; ++i) {
					int tmp_idx = idx + results[i].index;
					ld = nv_nn_dist(tree->mat[y + 1][tmp_idx],
									vec, vec_j);
					if (ld.f < min_dist) {
						min_x = results[i].index;
						min_dist = ld.f;
						min_index = ld.i;
					}
				}
				idx += min_x;
				idx *= tree->dim[y + 1];
				idx += min_index;
				nv_free(results);
				ld.i = idx;
				ld.f = min_dist;
			}
			break;
		} else {
			ld = nv_nn_dist(tree->mat[y][idx], vec, vec_j);
			x = ld.i;
			idx *= tree->dim[y];
			idx += x;
			ld.i = idx;
		}
	}
	
	return ld;
}


void
nv_pca_kmeans_tree_train(nv_pca_kmeans_tree_t *tree,
						 const nv_matrix_t *data,
						 int max_epoch)
{
	int y, x;
	int prev_width = 1;
	int *labels = nv_alloc_type(int, data->m);
	
	for (y = 0; y < tree->height; ++y) {
		long t = nv_clock();
		if (y == 0) {
			memset(labels, 0, sizeof(int) * data->m);
		} else {
			int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif			
			for (i = 0; i < data->m; ++i) {
				labels[i] = nv_pca_kmeans_tree_predict_label_ex(tree, y, data, i, NV_PCA_KMEANS_TREE_TOPN);
			}
		}
		if (nv_pca_kmeans_tree_progress_flag) {
			printf("nv_pca_kmeans_tree_train: vq: %ldms\n", nv_clock() - t);
			fflush(stdout);
		}
		
		for (x = 0; x < prev_width; ++x) {
			if (y == 0) {
				nv_pca_kmeans_tree_train_at(tree, data, y, x, max_epoch);
			} else {
				int i, j = 0;
				
				for (i = 0; i < data->m; ++i) {
					if (labels[i] == x) {
						j += 1;
					}
				}
				if (j != 0) {
					nv_matrix_t *data_part = nv_matrix_alloc(data->n, j);
					j = 0;
					for (i = 0; i < data->m; ++i) {
						if (labels[i] == x) {
							nv_vector_copy(data_part, j, data, i);
							j += 1;
						}
					}
					nv_pca_kmeans_tree_train_at(tree, data_part, y, x, max_epoch);
					nv_matrix_free(&data_part);
				} else {
					if (nv_pca_kmeans_tree_progress_flag) {
						printf("nv_pca_kmeans_tree_train: no data: y:%d, x:%d\n",
							   y, x);
						fflush(stdout);
					}
				}
			}
		}
		prev_width = tree->dim[y] * prev_width;
	}
}

#define NV_PCA_KMEANS_EIGEN_MAX_EPOCH 30

void
nv_pca_kmeans_tree_train_at(nv_pca_kmeans_tree_t *tree,
							const nv_matrix_t *data, int y, int x, int max_epoch)
{
	nv_matrix_t *mat = tree->mat[y][x];
	nv_matrix_t *eigen_vec = tree->eigen_vec[y][x];
	nv_matrix_t *labels = nv_matrix_alloc(1, data->m);
	nv_matrix_t *count = nv_matrix_alloc(1, mat->m);
	nv_matrix_t *data_pca = nv_matrix_alloc(tree->d, data->m);
	nv_cov_t *cov = nv_cov_alloc(tree->n);
	long t = nv_clock();
	int i;
	nv_matrix_zero(mat);
	nv_matrix_zero(count);
	nv_matrix_zero(labels);

	nv_cov_eigen_ex(cov, data, tree->d, NV_PCA_KMEANS_EIGEN_MAX_EPOCH);
	for (i = 0; i < tree->d; ++i) {
		nv_vector_copy(eigen_vec, i, cov->eigen_vec, i);
	}
	pca_projection_all(tree->eigen_vec[y][x], data_pca, data);
	//pca_projection_all(tree->eigen_vec[0][0], data_pca, data);
	nv_kmeans(mat, count, labels, data_pca, mat->m, max_epoch);
	if (nv_pca_kmeans_tree_progress_flag) {
		printf("nv_pca_kmeans_tree_train_at: (%d, %d), data: %d, %ldms\n",
			   y, x, data->m, nv_clock() - t);
		fflush(stdout);
	}
	nv_matrix_free(&labels);
	nv_matrix_free(&count);
	nv_matrix_free(&data_pca);
	nv_cov_free(&cov);
}

void
nv_pca_kmeans_tree_inherit_train(nv_pca_kmeans_tree_t *tree,
								 const nv_pca_kmeans_tree_t *base_tree,
								 const nv_matrix_t *data,
								 int max_epoch)
{
	int y, x;
	int prev_width = 1;
	int *labels = nv_alloc_type(int, data->m);
	int inherit = 1;
	
	NV_ASSERT(tree->n == base_tree->n);
	
	for (y = 0; y < tree->height; ++y) {
		long t = nv_clock();
		
		if (inherit) {
			if (y < base_tree->height &&
				y < tree->height &&
				base_tree->dim[y] == tree->dim[y])
			{
				for (x = 0; x < prev_width; ++x) {
					nv_matrix_copy_all(tree->mat[y][x], base_tree->mat[y][x]);
					nv_matrix_copy_all(tree->eigen_vec[y][x], base_tree->eigen_vec[y][x]);
				}
				if (nv_pca_kmeans_tree_progress_flag) {
					printf("nv_pca_kmeans_tree_train: inherit: (%d, 0-%d)\n",
						   y, prev_width);
					fflush(stdout);
				}
				prev_width = tree->dim[y] * prev_width;
				continue;
			} else {
				inherit = 0;
			}
		}
		
		if (y == 0) {
			memset(labels, 0, sizeof(int) * data->m);
		} else {
			int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif 
			for (i = 0; i < data->m; ++i) {
				labels[i] = nv_pca_kmeans_tree_predict_label_ex(tree, y, data, i, NV_PCA_KMEANS_TREE_TOPN);
			}
		}
		if (nv_pca_kmeans_tree_progress_flag) {
			printf("nv_pca_kmeans_tree_train: vq: %ldms\n", nv_clock() - t);
			fflush(stdout);
		}
		
		for (x = 0; x < prev_width; ++x) {
			if (y == 0) {
				nv_pca_kmeans_tree_train_at(tree, data, y, x, max_epoch);
			} else {
				int i, j = 0;
				
				for (i = 0; i < data->m; ++i) {
					if (labels[i] == x) {
						j += 1;
					}
				}
				if (j != 0) {
					nv_matrix_t *data_part = nv_matrix_alloc(data->n, j);
					j = 0;
					for (i = 0; i < data->m; ++i) {
						if (labels[i] == x) {
							nv_vector_copy(data_part, j, data, i);
							j += 1;
						}
					}
					nv_pca_kmeans_tree_train_at(tree, data_part, y, x, max_epoch);
					nv_matrix_free(&data_part);
				} else {
					if (nv_pca_kmeans_tree_progress_flag) {
						printf("nv_pca_kmeans_tree_train: no data: y:%d, x:%d\n",
							   y, x);
						fflush(stdout);
					}
				}
			}
		}
		prev_width = tree->dim[y] * prev_width;
	}
}

int
nv_pca_kmeans_tree_predict_label(const nv_pca_kmeans_tree_t *tree, const nv_matrix_t *vec, int vec_j)
{
	return nv_pca_kmeans_tree_predict_label_ex(tree, tree->height, vec, vec_j, NV_PCA_KMEANS_TREE_TOPN);
}

nv_int_float_t
nv_pca_kmeans_tree_predict_label_and_dist(const nv_pca_kmeans_tree_t *tree,
										  const nv_matrix_t *vec, int vec_j)
{
	int y;
	nv_int_float_t idx = {0, 0.0f};
	
	for (y = 0; y < tree->height; ++y) {
		nv_int_float_t x = nv_nn_dist(tree->mat[y][idx.i], vec, vec_j);
		if (y - 1 != tree->height) {
			idx.i *= tree->dim[y];
		}
		idx.i += x.i;
		idx.f = x.f;
	}
	
	return idx;
}


void
nv_pca_kmeans_tree_free(nv_pca_kmeans_tree_t **tree)
{
	if (tree && *tree) {
		int y, x;
		for (y = 0; y < (*tree)->height; ++y) {
			for (x = 0; x < (*tree)->node[y]; ++x) {
				nv_matrix_free(&(*tree)->mat[y][x]);
				nv_matrix_free(&(*tree)->eigen_vec[y][x]);
			}
			nv_free((*tree)->mat[y]);
			nv_free((*tree)->eigen_vec[y]);
		}
		nv_free((*tree)->mat);
		nv_free((*tree)->eigen_vec);
		nv_free((*tree)->dim);
		nv_free((*tree)->node);
		nv_free(*tree);
		*tree = NULL;
	}
}

void nv_pca_kmeans_tree_dump_c(FILE *out,
							   const nv_pca_kmeans_tree_t *tree,
							   const char *name, int static_variable)
{
	char var_name[1024];
	int y, x;
	
	for (y = 0; y < tree->height; ++y) {
		for (x = 0; x < tree->node[y]; ++x) {
			nv_snprintf(var_name, sizeof(var_name) - 1, "%s_%d_%d", name, y, x);
			nv_matrix_dump_c(out, tree->mat[y][x], var_name, 1);
		}
		fprintf(out, "static nv_matrix_t *%s_%d[] = {", name, y);
		for (x = 0; x < tree->node[y]; ++x) {
			if (x != 0) {
				fprintf(out, ", ");
				if (x % 8 == 0) {
					fprintf(out, "\n");
				}
			}
			fprintf(out, "&%s_%d_%d", name, y, x);
		}
		fprintf(out, "};\n");
	}
	fprintf(out, "static nv_matrix_t ** %s_mat[] = {", name);
	for (y = 0; y < tree->height; ++y) {
		if (y != 0) {
			fprintf(out, ", ");
			if (y % 8 == 0) {
				fprintf(out, "\n");
			}
		}
		fprintf(out, "%s_%d", name, y);
	}
	fprintf(out, "};\n");
			
	fprintf(out, "static int %s_dim[] = {", name);
	for (y = 0; y < tree->height; ++y) {
		if (y != 0) {
			fprintf(out, ", ");
			if (y % 8 == 0) {
				fprintf(out, "\n");
			}
		}
		fprintf(out, "%d", tree->dim[y]);
	}
	fprintf(out, "};\n");
	fprintf(out, "static int %s_node[] = {", name);
	for (y = 0; y < tree->height; ++y) {
		if (y != 0) {
			fprintf(out, ", ");
			if (y % 8 == 0) {
				fprintf(out, "\n");
			}
		}
		fprintf(out, "%d", tree->node[y]);
	}
	fprintf(out, "};\n");
	fprintf(out,
			"%s nv_pca_kmeans_tree_t %s = {\n %d, %d, %d, %s_dim, %s_node, %s_mat};",
			static_variable ? "static":"",
			name, tree->height, tree->n, tree->m,
			name, name, name);

	fflush(out);
}


void
nv_pca_kmeans_tree_label_vec(const nv_pca_kmeans_tree_t *tree,
							 nv_matrix_t **mat, int *vec_j,
							 int label)
{
	int y, x;
	
	y = tree->height - 1;
	x = label / tree->dim[y];
	*vec_j = label % tree->dim[y];
	*mat = tree->mat[y][x];
}
