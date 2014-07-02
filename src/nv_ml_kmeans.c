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
#include "nv_ml.h"
#include "nv_num.h"

/* k-means++ */

static int nv_kmeans_progress_flag = 0;
void nv_kmeans_progress(int onoff)
{
	nv_kmeans_progress_flag = onoff;
}

/* K-Means++初期値選択 */
void 
nv_kmeans_init_pp(nv_matrix_t *means, int k,
				  const nv_matrix_t *data, int tries)
{
	int local_tries = (tries > 0) ?
		tries :	NV_ROUND_INT(2.0f + logf((float)k) / logf(10.0f));
	nv_matrix_t *min_dists = nv_matrix_alloc(1, data->m);
	int m, c;
	float pot;
	int threads = nv_omp_procs();

	/* 1つ目 */
	nv_vector_copy(means, 0, data, nv_rand_index(data->m));

	pot = 0.0f;
	if (data->n == 3) {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:pot) num_threads(threads)
#endif
		for (m = 0; m < data->m; ++m) {
			float dist =
				(NV_MAT_V(means, 0, 0) - NV_MAT_V(data, m, 0)) *
				(NV_MAT_V(means, 0, 0) - NV_MAT_V(data, m, 0)) +
				(NV_MAT_V(means, 0, 1) - NV_MAT_V(data, m, 1)) *
				(NV_MAT_V(means, 0, 1) - NV_MAT_V(data, m, 1)) +
				(NV_MAT_V(means, 0, 2) - NV_MAT_V(data, m, 2)) *
				(NV_MAT_V(means, 0, 2) - NV_MAT_V(data, m, 2));
			NV_MAT_V(min_dists, m, 0) = dist;
			pot += dist;
		}
	} else {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:pot) num_threads(threads)
#endif
		for (m = 0; m < data->m; ++m) {
			float dist = nv_euclidean2(means, 0, data, m);
			NV_MAT_V(min_dists, m, 0) = dist;
			pot += dist;
		}
	}

	for (c = 1; c < k; ++c) {
		int i;
		int best_index = -1;
		float min_pot = FLT_MAX;

		for (i = 0; i < local_tries; ++i) {
			float new_pot;
			float limit = pot * nv_rand();
			int j, l;

			for (l = 0; l < data->m - 1; ++l) {
				float min_dist = NV_MAT_V(min_dists, l, 0);
				if (limit < min_dist) {
					break;
				} else {
					limit -= min_dist;
				}
			}
			new_pot = 0.0;
			if (data->n == 3) {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:new_pot) num_threads(threads)
#endif
				for (j = 0; j < data->m; ++j) {
					float min_dist = NV_MAT_V(min_dists, j, 0);
					float dist =
						(NV_MAT_V(data, l, 0) - NV_MAT_V(data, j, 0)) *
						(NV_MAT_V(data, l, 0) - NV_MAT_V(data, j, 0)) +
						(NV_MAT_V(data, l, 1) - NV_MAT_V(data, j, 1)) *
						(NV_MAT_V(data, l, 1) - NV_MAT_V(data, j, 1)) +
						(NV_MAT_V(data, l, 2) - NV_MAT_V(data, j, 2)) *
						(NV_MAT_V(data, l, 2) - NV_MAT_V(data, j, 2));
					new_pot += NV_MIN(dist, min_dist);
				}
			} else {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:new_pot) num_threads(threads)
#endif
				for (j = 0; j < data->m; ++j) {
					float dist = nv_euclidean2(data, l, data, j);
					float min_dist = NV_MAT_V(min_dists, j, 0);
					new_pot += NV_MIN(dist, min_dist);
				}
			}
			if (new_pot < min_pot) {
				min_pot = new_pot;
				best_index = l;
			}
		}
		
		nv_vector_copy(means, c, data, best_index);
		pot = min_pot;
		
		if (c - 1 < k) {
			if (data->n == 3) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
				for (i = 0; i < data->m; ++i) {
					float dist =
						(NV_MAT_V(data, i, 0) - NV_MAT_V(data, best_index, 0)) *
						(NV_MAT_V(data, i, 0) - NV_MAT_V(data, best_index, 0)) +
						(NV_MAT_V(data, i, 1) - NV_MAT_V(data, best_index, 1)) *
						(NV_MAT_V(data, i, 1) - NV_MAT_V(data, best_index, 1)) +
						(NV_MAT_V(data, i, 2) - NV_MAT_V(data, best_index, 2)) *
						(NV_MAT_V(data, i, 2) - NV_MAT_V(data, best_index, 2));
					if (dist < NV_MAT_V(min_dists, i, 0)) {
						NV_MAT_V(min_dists, i, 0) = dist;
					}
				}
			} else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
				for (i = 0; i < data->m; ++i) {
					float dist = nv_euclidean2(data, i, data, best_index);
					if (dist < NV_MAT_V(min_dists, i, 0)) {
						NV_MAT_V(min_dists, i, 0) = dist;
					}
				}
			}
		}
	}

	nv_matrix_free(&min_dists);
}

/* 選択済みクラスから一番距離が遠いクラスを選ぶ初期値選択 */
void 
nv_kmeans_init_dist(nv_matrix_t *means, int k,
					const nv_matrix_t *data)
{
	nv_matrix_t *dists = nv_matrix_alloc(data->m, k);	
	int i, j;
	int threads = nv_omp_procs();

	if (data->m == 0) {
		nv_matrix_free(&dists);
		return;
	}
	nv_vector_copy(means, 0, data, nv_rand_index(data->m));
	j = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
	for (i = 0; i < data->m; ++i) {
		NV_MAT_V(dists, j, i) = nv_euclidean2(means, j, data, i);
	}
	
	for (j = 0; j < k - 1; ++j) {
		float max_v = -FLT_MAX;
		int max_i = -1;

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
		for (i = 0; i < data->m; ++i) {
			NV_MAT_V(dists, j, i) = nv_euclidean2(means, j, data, i);
		}
		for (i = 0; i < data->m; ++i) {
			float min_dist = FLT_MAX;
			int l;
			for (l = 0; l <= j; ++l) {
				if (min_dist > NV_MAT_V(dists, l, i)) {
					min_dist = NV_MAT_V(dists, l, i);
				}
			}
			if (min_dist > max_v) {
				max_v = min_dist;
				max_i = i;
			}
		}
		nv_vector_copy(means, j + 1, data, max_i);
	}
	nv_matrix_free(&dists);
}

void
nv_kmeans_init_rand(nv_matrix_t *means)
{
	int j;
	for (j = 0; j < means->m; ++j) {
		nv_vector_nrand(means, j, 0.0f, 0.1f);
	}
}

/* 最小距離クラス選択 */
static int 
nv_kmeans_bmc(const nv_matrix_t *mat, int mk,
			  const nv_matrix_t *vec, int vm)
{
	int k;
	int min_k = -1;
	float min_dist = FLT_MAX;
	
	if (mat->n == 3) {
		/* color vec specialize */
		for (k = 0; k < mk; ++k) {
			float dist =
				(NV_MAT_V(vec, vm, 0) - NV_MAT_V(mat, k, 0)) *
				(NV_MAT_V(vec, vm, 0) - NV_MAT_V(mat, k, 0)) +
				(NV_MAT_V(vec, vm, 1) - NV_MAT_V(mat, k, 1)) *
				(NV_MAT_V(vec, vm, 1) - NV_MAT_V(mat, k, 1)) +
				(NV_MAT_V(vec, vm, 2) - NV_MAT_V(mat, k, 2)) *
				(NV_MAT_V(vec, vm, 2) - NV_MAT_V(mat, k, 2));
			if (dist < min_dist) {
				min_dist = dist;
				min_k = k;
			}
		}
	} else {
		for (k = 0; k < mk; ++k) {
			float dist = nv_euclidean2(mat, k, vec, vm);
			if (dist < min_dist) {
				min_dist = dist;
				min_k = k;
			}
		}
	}
	NV_ASSERT(min_k != -1);

	return min_k;
}

int 
nv_kmeans_em(nv_matrix_t *means,  // k
			 nv_matrix_t *count,  // k
			 nv_matrix_t *labels_, // data->m
			 const nv_matrix_t *data,
			 const int k,
			 const int max_epoch)
{
	int j, l;
	int processing = 1;
	int converge, epoch;
	long t = nv_clock();
	int threads = nv_omp_procs();
	
	nv_matrix_t *labels = nv_matrix_alloc(data->m, 1);
	nv_matrix_t *old_labels = nv_matrix_alloc(data->m, 1);
	nv_matrix_t *sum = nv_matrix_alloc(data->n, k);
	nv_matrix_t *sum_tmp = nv_matrix_list_alloc(data->n, k, threads);
	nv_matrix_t *count_tmp = nv_matrix_alloc(k, threads);

	NV_ASSERT(means->n == data->n);
	NV_ASSERT(means->m >= k);
	NV_ASSERT(labels_->m >= data->m);

	nv_matrix_fill(old_labels, -1.0f);
	for (j = 0; j < data->m; ++j) {
		NV_MAT_V(labels, 0, j) = NV_MAT_V(labels_, j, 0);
	}
	
	epoch = 0;
	do {
		if (nv_kmeans_progress_flag) {
			printf("nv_kmeans: epoch: %d, %ldms\n", epoch, nv_clock() - t);
			fflush(stdout);
		}
		t = nv_clock();
		nv_matrix_zero(count_tmp);
		nv_matrix_zero(count);
		nv_matrix_zero(sum_tmp);
		nv_matrix_zero(sum);
		
		/* ラベルを割り当て,各クラスの合計ベクトルを計算 */
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
		for (j = 0; j < data->m; ++j) {
			int label = nv_kmeans_bmc(means, k, data, j);
			int thread_idx = nv_omp_thread_id();
			int i;
			
			NV_MAT_V(labels, 0, j) = (float)label;
			NV_MAT_V(count_tmp, thread_idx, label) += 1.0f;
			for (i = 0; i < means->n; ++i) {
				NV_MAT_LIST_V(sum_tmp, thread_idx, label, i) += NV_MAT_V(data, j, i);
			}
		}
		for (l = 0; l < threads; ++l) {
			int i;
			for (j = 0; j < k; ++j) {
				NV_MAT_V(count, j, 0) += NV_MAT_V(count_tmp, l, j);
				for (i = 0; i < sum->n; ++i) {
					NV_MAT_V(sum, j, i) += NV_MAT_LIST_V(sum_tmp, l, j, i);
				}
			}
		}
		++epoch;
		
		/* 終了判定 */
		converge = nv_vector_eq(labels, 0, old_labels, 0);
		if (converge) {
			/* 終了 */
			processing = 0;
		} else {
			/* ラベル更新 */
			nv_vector_copy(old_labels, 0, labels, 0);
			
			/* 平均値計算 */
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
			for (j = 0; j < k; ++j) {
				if (NV_MAT_V(count, j, 0) != 0.0f) {
					nv_vector_muls(means, j, sum, j, 1.0f / NV_MAT_V(count, j, 0));
				} else {
					if (nv_kmeans_progress_flag) {
						printf("nv_kmeans: epoch: empty class found\n");
						fflush(stdout);
					}
				}
			}
			
			/* 最大試行回数判定 */
			if (max_epoch != 0
				&& epoch >= max_epoch)
			{
				/* 終了 */
				processing = 0;
			}
		}
	} while (processing);
	
	for (j = 0; j < data->m; ++j) {
		NV_MAT_V(labels_, j, 0) = NV_MAT_V(labels, 0, j);
	}
	
	nv_matrix_free(&old_labels);
	nv_matrix_free(&labels);
	nv_matrix_free(&sum);
	nv_matrix_free(&sum_tmp);
	nv_matrix_free(&count_tmp);
	
	return k;
}

int 
nv_kmeans(nv_matrix_t *means,  // k
		  nv_matrix_t *count,  // k
		  nv_matrix_t *labels, // data->m
		  const nv_matrix_t *data,
		  const int k,
		  const int max_epoch)
{
	int ret;
	long t = nv_clock();
	
	/* 初期値選択 */
	nv_kmeans_init_pp(means, k, data, -1);
	
	if (nv_kmeans_progress_flag) {
		printf("nv_kmeans: init, %ldms\n", nv_clock() - t);
		fflush(stdout);
	}
	/* EM */
	ret = nv_kmeans_em(means, count, labels, data, k, max_epoch);
	if (nv_kmeans_progress_flag) {
		printf("nv_kmeans: kmeans, %ldms\n", nv_clock() - t);
		fflush(stdout);
	}

	return ret;
}

