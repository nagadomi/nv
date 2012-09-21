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

/* X-Means
 * 分割の基準が難しくてあまりうまく動かない
 */

#include "nv_core.h"
#include "nv_num.h"
#include "nv_ml_kmeans.h"
#include "nv_ml_xmeans.h"
#include "nv_ml_gaussian.h"

typedef struct {
	nv_matrix_t *data;
	nv_matrix_t *mean;
	int *link;
} nv_xmeans_stack_elm_t;


static nv_xmeans_stack_elm_t *
nv_xmeans_stack_push(nv_xmeans_stack_elm_t **stack, int *i,
					 const nv_matrix_t *data,
					 const nv_matrix_t *labels,
					 const nv_matrix_t *mean,
					 float label)
{
	nv_xmeans_stack_elm_t *elm = (nv_xmeans_stack_elm_t *)
		                       malloc(sizeof(nv_xmeans_stack_elm_t));
	int m, dm = 0;
	
	elm->data = nv_matrix_alloc(data->n, data->m);
	elm->mean = nv_matrix_alloc(mean->n, 1);
	elm->link = (int *)malloc(sizeof(int) * data->m);

	nv_vector_copy(elm->mean, 0, mean, (int)label);

	for (m = 0; m < data->m; ++m) {
		if (NV_MAT_V(labels, m, 0) == label) {
			nv_vector_copy(elm->data, dm, data, m);
			elm->link[dm] = m;
			++dm;
		}
	}
	elm->data->m = dm;
	stack[*i] = elm;
	++*i;

	return elm;
}

static nv_xmeans_stack_elm_t *
nv_xmeans_stack_pop(nv_xmeans_stack_elm_t **stack, int *i)
{
	if (*i == 0) {
		return NULL;
	}

	return stack[--*i];
}

static void 
nv_xmeans_stack_elm_free(nv_xmeans_stack_elm_t *elm)
{
	nv_matrix_free(&elm->data);
	nv_matrix_free(&elm->mean);
	free(elm->link);
	free(elm);
}
#if 0
static float 
nv_xmeans_log_likelihood(const nv_matrix_t *cls)
{
	nv_cov_t *cov = nv_cov_alloc(cls->n);
	float likelihood = 0.0f;
	int m;

	nv_cov_eigen(cov, cls);
	for (m = 0; m < cls->m; ++m) {
		likelihood += nv_gaussian_log_predict(0, cov, cls, m);
	}
	nv_cov_free(&cov);

	return likelihood;
}
#endif
static float 
nv_xmeans_variance(const nv_matrix_t *cls, float c, const nv_matrix_t *means, int k)
{
	int m;
	float variance = 0.0f;
	float factor;
	
	if (cls->m > c) {
		factor = 1.0f / (cls->m - c);
		for (m = 0; m < cls->m; ++m) {
			float dist = nv_euclidean(means, k, cls, m);
			variance += factor * dist;
		}
	}
	return variance;
}

#if 0
static float 
nv_xmeans_var(const nv_matrix_t *cls, const nv_matrix_t *means, int k)
{
	int m;
	float dist_max = -FLT_MAX;
	float ratio = 0.0f;
	for (m = 0; m < cls->m; ++m) {
		dist_max = NV_MAX(dist_max, nv_euclidean(cls, m, means, k));
	}
	dist_max *= 0.8f;
	for (m = 0; m < cls->m; ++m) {
		ratio += nv_euclidean(cls, m, means, k) / dist_max;
	}
	ratio *= 1.0f / cls->m;
	return -ratio;
}
#endif

#if 0
static float 
nv_xmeans_var2(const nv_matrix_t *cls1, const nv_matrix_t *cls2, 
			   const nv_matrix_t *means, int k1, int k2)
{
	float var = NV_MIN(nv_xmeans_var(cls1, means, k1), nv_xmeans_var(cls2, means, k2));
	return var;
}
#endif

static float 
nv_xmeans_bic(const nv_matrix_t *cls, const nv_matrix_t *means, int k)
{
	float variance = nv_xmeans_variance(cls, 1.0f, means, k);
	float r = (float)cls->m;
	float n = (float)cls->n;
	float c = 1.0f;
	float p = (c - 1.0f) + c * n + c;

	float likelihood = variance == 0.0f ? 0.0f:
		- ((r / 2.0f) * nv_log2(2.0f * NV_PI))
		- (((r * n) / 2.0f) * nv_log2(variance))
		- ((r - c) / 2.0f)
		+ (r * nv_log2(r))
		- (r * nv_log2(r));

	float bic = -1.0f * likelihood + ((p / 2.0f) * (nv_log2(r)));
	/* float bic = -2.0f * likelihood + 2.0f * p; */

	return bic;
}

static float 
nv_xmeans_bic2(const nv_matrix_t *cls1, const nv_matrix_t *cls2, 
			   const nv_matrix_t *means, int k1, int k2)
{
	float variance1 = nv_xmeans_variance(cls1, 2.0f, means, k1);
	float variance2 = nv_xmeans_variance(cls2, 2.0f, means, k2);
	float r1 = (float)cls1->m;
	float r2 = (float)cls2->m;
	float n = (float)cls1->n;
	float c = 2.0f;
	float p = (c - 1.0f) + c * n + c;

	float likelihood1 = variance1 == 0.0f ? 0.0f:
		  - ((r1 / 2.0f) * nv_log2(2.0f * NV_PI))
		  - (((r1 * n) / 2.0f) * nv_log2(variance1))
		  - ((r1 - c) / 2.0f)
		  + (r1 * nv_log2(r1))
		  - (r1 * nv_log2(r1 + r2));
   float likelihood2 = variance2 == 0.0f ? 0.0f:
		  - ((r2 / 2.0f) * nv_log2(2.0f * NV_PI))
		  - (((r2 * n) / 2.0f) * nv_log2(variance2))
		  - ((r2 - c) / 2.0f)
		  + (r2 * nv_log2(r2))
		  - (r2 * nv_log2(r1 + r2));
   float likelihood = likelihood1 + likelihood2;

   float bic = -1.0f * likelihood + ((p / 2.0f) * (nv_log2(r1 + r2)));
   /* float bic = -2.0f * likelihood + 2.0f * p; */
   return bic;
}

int nv_xmeans(nv_matrix_t *means,
			  nv_matrix_t *count,
			  nv_matrix_t *labels,
			  const nv_matrix_t *data,
			  const int max_k, /* 2n */
			  const int max_epoch)
{
	int k = 2;
	int m, n;
	int stack_i = 0;
	int label_max = 0;
	nv_xmeans_stack_elm_t *elm;
	nv_xmeans_stack_elm_t **stack = (nv_xmeans_stack_elm_t **)
		                         malloc(sizeof(nv_xmeans_stack_elm_t *) * max_k * 2);

	/* (^^ */
	nv_kmeans(means, count, labels, data, 1, max_epoch);
	nv_xmeans_stack_push(
		stack, &stack_i,
		data, labels, means, 0.0f
    );

	/* 初期化 */
	nv_matrix_zero(labels);
	nv_matrix_zero(count);
	nv_matrix_zero(means);

	/* 幅優先で分割 */
	while ((elm = nv_xmeans_stack_pop(stack, &stack_i))) {
		float old_bic = nv_xmeans_bic(elm->data, elm->mean, 0);
		nv_matrix_t *cur_labels = nv_matrix_alloc(1, elm->data->m);
		nv_matrix_t *cur_count = nv_matrix_alloc(1, 2);
		nv_matrix_t *cur_means = nv_matrix_alloc(elm->data->n, 2);
		float new_bic;
		nv_xmeans_stack_elm_t *new_class[2];

		nv_kmeans(cur_means, cur_count, cur_labels, elm->data, 2, max_epoch);
		new_class[0] = nv_xmeans_stack_push(
			stack, &stack_i,
			elm->data, cur_labels, cur_means, 0.0f
		);
		new_class[1] = nv_xmeans_stack_push(
			stack, &stack_i,
			elm->data, cur_labels, cur_means, 1.0f
		);

		new_bic = nv_xmeans_bic2(new_class[0]->data, new_class[1]->data, cur_means, 0, 1);
		if (new_bic < old_bic) {
			/* 分割あり */
			int i;
			for (i = 0; i < 2; ++i) {
				/* リンク更新 */
				for (m = 0; m < new_class[i]->data->m; ++m) {
					new_class[i]->link[m] = elm->link[new_class[i]->link[m]];
				}
				/* ラベル更新 */
				if (i != 0) {
					++label_max;
					for (m = 0; m < new_class[i]->data->m; ++m) {
						NV_MAT_V(labels, new_class[i]->link[m], 0) = (float)label_max;
					}
				}
			}
		} else {
			/* 分割なし */
			new_class[1] = nv_xmeans_stack_pop(stack, &stack_i);
			new_class[0] = nv_xmeans_stack_pop(stack, &stack_i);
			nv_xmeans_stack_elm_free(new_class[1]);
			nv_xmeans_stack_elm_free(new_class[0]);
		}
		nv_matrix_free(&cur_labels);
		nv_matrix_free(&cur_count);
		nv_matrix_free(&cur_means);
		nv_xmeans_stack_elm_free(elm);

		/* 最大数 */
		if (max_k && label_max >= max_k - 1) {
			break;
		}
	}
	while ((elm = nv_xmeans_stack_pop(stack, &stack_i))) {
		nv_xmeans_stack_elm_free(elm);
	}
	free(stack);

	/* 最終ラベル数 */
	label_max += 1;

	/* 中央値更新 */
	for (m = 0; m < data->m; ++m) {
		int label = (int)NV_MAT_V(labels, m, 0);
		for (n = 0; n < data->n; ++n) {
			NV_MAT_V(means, label, n) += NV_MAT_V(data, m, n);
		}
		NV_MAT_V(count, label, 0) += 1.0f;
	}
	for (k = 0; k < label_max; ++k) {
		if (NV_MAT_V(count, k, 0) != 0.0f) {
			float factor = 1.0f / NV_MAT_V(count, k, 0);
			for (n = 0; n < means->n; ++n) {
				NV_MAT_V(means, k, n) *= factor;
			}
		}
	}

	return label_max;
}
