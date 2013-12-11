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

static int nv_klr_tree_progress_flag = 0;

void
nv_klr_tree_progress(int flag)
{
	nv_klr_tree_progress_flag = flag;
}

nv_klr_tree_t *
nv_klr_tree_alloc(int n, int *dim, int height)
{
	nv_klr_tree_t *tree = nv_alloc_type(nv_klr_tree_t, 1);
	int y, x;
	int m = 1;
	int prev_width = 1;
	
	tree->height = height;
	tree->dim = nv_alloc_type(int, height);
	tree->node = nv_alloc_type(int, height);
	tree->lr = nv_alloc_type(nv_lr_t **, tree->height);
	
	for (y = 0; y < height; ++y) {
		tree->dim[y] = dim[y];
		tree->node[y] = prev_width;
		tree->lr[y] = nv_alloc_type(nv_lr_t *, prev_width + 1);
		
		for (x = 0; x < prev_width; ++x) {
			tree->lr[y][x] = nv_lr_alloc(n, dim[y]);
		}
		
		prev_width = dim[y] * prev_width;
		m *= dim[y];
	}
	tree->n = n;
	tree->m = m;
	if (nv_klr_tree_progress_flag) {
		printf("nv_klr_tree_alloc: alloc: %d dim\n", tree->m);
		fflush(stdout);		
	}
	
	return tree;
}

int
nv_klr_tree_predict_label_ex(const nv_klr_tree_t *tree, int height,
							 const nv_matrix_t *vec, int vec_j)
{
	int y, x;
	int idx = 0;
	
	for (y = 0; y < height; ++y) {
		x = nv_lr_predict_label(tree->lr[y][idx], vec, vec_j);
		idx *= tree->dim[y];
		idx += x;
	}
	
	return idx;
}

int
nv_klr_tree_predict_label(const nv_klr_tree_t *tree, const nv_matrix_t *vec, int vec_j)
{
	return nv_klr_tree_predict_label_ex(tree, tree->height, vec, vec_j);
}

void
nv_klr_tree_train(nv_klr_tree_t *tree,
				  const nv_matrix_t *data,
				  nv_lr_param_t param,
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
				labels[i] = nv_klr_tree_predict_label_ex(tree, y, data, i);
			}
		}
		if (nv_klr_tree_progress_flag) {
			printf("nv_klr_tree_train: vq: %ldms\n", nv_clock() - t);
			fflush(stdout);
		}
		
		for (x = 0; x < prev_width; ++x) {
			if (y == 0) {
				nv_klr_tree_train_at(tree, data, y, x, param, max_epoch);
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
					nv_klr_tree_train_at(tree, data_part, y, x, param, max_epoch);
					nv_matrix_free(&data_part);
				} else {
					if (nv_klr_tree_progress_flag) {
						printf("nv_klr_tree_train: no data: y:%d, x:%d\n",
							   y, x);
						fflush(stdout);
					}
				}
			}
		}
		prev_width = tree->dim[y] * prev_width;
	}
}

void
nv_klr_tree_train_at(nv_klr_tree_t *tree,
					 const nv_matrix_t *data, int y, int x,
					 nv_lr_param_t param,
					 int max_epoch)
{
	nv_lr_t *lr = tree->lr[y][x];
	nv_matrix_t *labels = nv_matrix_alloc(1, data->m);
	nv_matrix_t *count = nv_matrix_alloc(1, lr->k);
	long t = nv_clock();
	
	nv_klr_init(lr, count, labels, data,
				NV_LR_PARAM(10,
							param.grad_w,
							param.reg_type, param.reg_w, 1));
	nv_klr_em(lr, count, labels, data, param, max_epoch);
	nv_lr_train(lr, data, labels,
				NV_LR_PARAM(20,
							param.grad_w,
							param.reg_type, param.reg_w, 1));
	
	if (nv_klr_tree_progress_flag) {
		printf("nv_klr_tree_train_at: (%d, %d), data: %d, %ldms\n",
			   y, x, data->m, nv_clock() - t);
		fflush(stdout);
	}
	
	nv_matrix_free(&labels);
	nv_matrix_free(&count);
}

void
nv_klr_tree_inherit_train(nv_klr_tree_t *tree,
						  const nv_klr_tree_t *base_tree,
						  const nv_matrix_t *data,
						  nv_lr_param_t param,
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
					nv_matrix_copy_all(tree->lr[y][x]->w, base_tree->lr[y][x]->w);
				}
				if (nv_klr_tree_progress_flag) {
					printf("nv_klr_tree_train: inherit: (%d, 0-%d)\n",
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
				labels[i] = nv_klr_tree_predict_label_ex(tree, y, data, i);
			}
		}
		if (nv_klr_tree_progress_flag) {
			printf("nv_klr_tree_train: vq: %ldms\n", nv_clock() - t);
			fflush(stdout);
		}
		
		for (x = 0; x < prev_width; ++x) {
			if (y == 0) {
				nv_klr_tree_train_at(tree, data, y, x, param, max_epoch);
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
					nv_klr_tree_train_at(tree, data_part, y, x, param, max_epoch);
					nv_matrix_free(&data_part);
				} else {
					if (nv_klr_tree_progress_flag) {
						printf("nv_klr_tree_train: no data: y:%d, x:%d\n",
							   y, x);
						fflush(stdout);
					}
				}
			}
		}
		prev_width = tree->dim[y] * prev_width;
	}
}

void
nv_klr_tree_free(nv_klr_tree_t **tree)
{
	if (tree && *tree) {
		int y, x;
		for (y = 0; y < (*tree)->height; ++y) {
			for (x = 0; x < (*tree)->node[y]; ++x) {
				nv_lr_free(&(*tree)->lr[y][x]);
			}
			nv_free((*tree)->lr[y]);
		}
		nv_free((*tree)->lr);
		nv_free((*tree)->dim);
		nv_free((*tree)->node);
		nv_free(*tree);
		*tree = NULL;
	}
}

void
nv_klr_tree_dump_c(FILE *out,
				   const nv_klr_tree_t *tree,
				   const char *name,
				   int static_variable)
{
	char var_name[1024];
	int y, x;
	
	for (y = 0; y < tree->height; ++y) {
		for (x = 0; x < tree->node[y]; ++x) {
			nv_snprintf(var_name, sizeof(var_name) - 1, "%s_%d_%d", name, y, x);
			nv_lr_dump_c(out, tree->lr[y][x], var_name, 1);
		}
		fprintf(out, "static nv_lr_t *%s_%d[] = {", name, y);
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
	fprintf(out, "static nv_lr_t ** %s_lr[] = {", name);
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
			"%snv_klr_tree_t %s = {\n %d, %d, %d, %s_dim, %s_node, %s_lr};\n",
			static_variable ? "static ":"",
			name, tree->height, tree->n, tree->m,
			name, name, name);

	fflush(out);
}

int
nv_klr_tree_predict(const nv_klr_tree_t *tree,
					nv_int_float_t *lp,
					const nv_matrix_t *vec, int vec_j)
{
	int y;
	int idx = 0;

	for (y = 0; y < tree->height; ++y) {
		lp[y] = nv_lr_predict_label_and_probability(tree->lr[y][idx], vec, vec_j);
		idx *= tree->dim[y];
		idx += lp[y].i;
	}
	
	return idx;
}

int
nv_klr_tree_leaf_vector(const nv_klr_tree_t *tree,
						nv_matrix_t *prob, int pj,
						int *offset,
						const nv_matrix_t *vec, int vec_j)
{
	int y;
	int idx = 0;
	nv_int_float_t lp;
	float p = 1.0f;
	int label = -1;
	float max_p = -FLT_MAX;
	
	NV_ASSERT(prob->n == tree->dim[tree->height - 1]);
	
	for (y = 0; y < tree->height; ++y) {
		if (y != tree->height - 1) {
			lp = nv_lr_predict_label_and_probability(tree->lr[y][idx], vec, vec_j);
			p *= lp.f;
			idx *= tree->dim[y];
			idx += lp.i;
		} else {
			int k = tree->lr[y][idx]->k;
			int i;

			*offset = tree->lr[y][idx]->k * idx;
			
			nv_lr_predict_vector(tree->lr[y][idx], prob, pj, vec, vec_j);
			
			for (i = 0; i < k; ++i) {
				if (NV_MAT_V(prob, pj, i) > max_p) {
					max_p = NV_MAT_V(prob, pj, i);
					label = i;
				}
			}
			label += *offset;
		}
	}
	
	return label;
}

void
nv_klr_tree_predict_vector(const nv_klr_tree_t *tree,
						   nv_matrix_t *prob, int pj,
						   const nv_matrix_t *vec, int vec_j)
{
	int y = tree->height - 1;
	int k = tree->lr[y][0]->k;
	int i;
	
	NV_ASSERT(prob->n == k * tree->node[y]);

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (i = 0; i < tree->node[y]; ++i) {
		nv_matrix_t *v = nv_matrix_alloc(tree->lr[y][i]->k, 1);
		nv_lr_predict_vector(tree->lr[y][i], v, 0, vec, vec_j);
		memmove(&NV_MAT_V(prob, pj, i * k), v->v, sizeof(float) * v->step);
		nv_matrix_free(&v);
	}
}


