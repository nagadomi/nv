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

static int nv_lbgu_progress_flag = 0;

void nv_lbgu_progress(int onoff)
{
	nv_lbgu_progress_flag = onoff;
}

/* LBG-U, 初期化+実行 */
int 
nv_lbgu(nv_matrix_t *means,  // k
		nv_matrix_t *count,  // k
		nv_matrix_t *labels, // data->m
		const nv_matrix_t *data,
		const int k,
		const int kmeans_max_epoch,
		const int max_epoch)
{
	int ret;

	nv_kmeans_init_pp(means, k, data, -1);
	ret = nv_lbgu_em(means, count, labels, data, k, kmeans_max_epoch, max_epoch);

	return ret;
}

static float
nv_lbgu_e(nv_matrix_t *e,
		  const nv_matrix_t *means,
		  const nv_matrix_t *data,
		  const nv_matrix_t *labels,
		  const nv_matrix_t *count)
{
	int m;
	float rmse = 0.0f;

	nv_matrix_zero(e);

	for (m = 0; m < data->m; ++m) {
		int i = NV_MAT_VI(labels, m, 0);
		float dist = nv_euclidean2(means, i, data, m);

		{
			NV_MAT_V(e, i, 0) += dist;
			rmse += dist;
		}
	}

	return rmse / data->m;
}

static void 
nv_lbgu_u(nv_matrix_t *u,
		  const nv_matrix_t *means,
		  const nv_matrix_t *data,
		  const nv_matrix_t *labels,
		  const nv_matrix_t *count)
{
	int m;
	nv_matrix_t *scale = nv_matrix_alloc(1, count->m);

	nv_matrix_zero(u);

	for (m = 0; m < count->m; ++m) {
		NV_MAT_V(scale, m, 0) = 1.0f / NV_MAT_V(count, m, 0);
	}

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
	for (m = 0; m < data->m; ++m) {
		int k;
		float diff, min_error = FLT_MAX;
		int i = NV_MAT_VI(labels, m, 0);

		for (k = 0; k < means->m; ++k) {
			float dist;
			if (k == i) {
				continue;
			}
			dist = nv_euclidean2(means, k, data, m);
			if (min_error > dist) {
				min_error = dist;
			}
		}
		diff = min_error - nv_euclidean2(means, i, data, m);

#ifdef _OPENMP
#pragma omp critical (nv_lbgu_u)
#endif
		{
			NV_MAT_V(u, i, 0) += diff;
		}
	}

	nv_matrix_free(&scale);
}

static void 
nv_lbgu_update(nv_matrix_t *means,
			   const nv_matrix_t *data,
			   const nv_matrix_t *labels,
			   const nv_matrix_t *count,
			   int max_error_class,
			   int min_error_class,
			   int kmeans_max_epoch)
{
	/*
	 * 分割対象のクラスタを2つにクラスタリングしてそのセントロイドで更新する
	 * (Bernd Fritzkeの論文とは異なる実装)
	 */

	int m, j;
	int c = NV_MAT_VI(count, max_error_class, 0);
	nv_matrix_t *data_tmp = nv_matrix_alloc(means->n, c);
	nv_matrix_t *means_tmp = nv_matrix_alloc(means->n, 2);
	nv_matrix_t *labels_tmp = nv_matrix_alloc(1, c);
	nv_matrix_t *count_tmp = nv_matrix_alloc(1, 2);

	nv_matrix_zero(data_tmp);

	for (m = j = 0; m < data->m; ++m) {
		if (max_error_class == NV_MAT_VI(labels, m, 0)) {
			nv_vector_copy(data_tmp, j++, data, m);
		}
	}
	NV_ASSERT(j == c);

	nv_kmeans(means_tmp, count_tmp, labels_tmp, data_tmp, 2, kmeans_max_epoch);
	nv_vector_copy(means, max_error_class, means_tmp, 0);
	nv_vector_copy(means, min_error_class, means_tmp, 1);

	nv_matrix_free(&data_tmp);
	nv_matrix_free(&means_tmp);
	nv_matrix_free(&labels_tmp);
	nv_matrix_free(&count_tmp);
}

int 
nv_lbgu_em(nv_matrix_t *means, // k
		   nv_matrix_t *count,  // k
		   nv_matrix_t *labels, // data->m
		   const nv_matrix_t *data,
		   const int k,
		   const int kmeans_max_epoch,
		   const int max_epoch)
{
	int m, epoch, max_error_class, min_error_class;
	float min_error, max_error, error_active, error_best = FLT_MAX;
	nv_matrix_t *e = nv_matrix_alloc(1, k);
	nv_matrix_t *u = nv_matrix_alloc(1, k);
	nv_matrix_t *means_tmp = nv_matrix_alloc(means->n, means->m);
	nv_matrix_t *count_tmp = nv_matrix_alloc(count->n, count->m);
	nv_matrix_t *labels_tmp = nv_matrix_alloc(labels->n, labels->m);
	long t;

	NV_ASSERT(means->n == data->n);
	NV_ASSERT(means->m >= k);
	NV_ASSERT(labels->m >= data->m);

	nv_matrix_copy(means_tmp, 0, means, 0, means_tmp->m);

	epoch = 0;
	while(1) 
	{
		t = nv_clock();
		/* k-means(LBG) */
		nv_kmeans_em(means_tmp, count_tmp, labels_tmp, data, k, kmeans_max_epoch);

		/* エラーが最大のクラスを選択 */
		error_active = nv_lbgu_e(e, means_tmp, data, labels_tmp, count_tmp);
		max_error_class = -1;
		max_error = -FLT_MAX;
		for (m = 0; m < k; ++m) {
			if (max_error < NV_MAT_V(e, m, 0)) {
				max_error = NV_MAT_V(e, m, 0);
				max_error_class = m;
			}
		}

		if (error_active >= error_best) {
			/* 前回よりエラーが減らなければ終了 */
			break;
		} else {
			error_best = error_active;
		}
		nv_matrix_copy(means, 0, means_tmp, 0, means_tmp->m);
		nv_matrix_copy(count, 0, count_tmp, 0, count_tmp->m);
		nv_matrix_copy(labels, 0, labels_tmp, 0, labels_tmp->m);

		if (++epoch >= max_epoch) {
			break;
		}

		/* エラー最小化の貢献が最小のクラスを選択 */
		nv_lbgu_u(u, means_tmp, data, labels_tmp, count_tmp);
		min_error = FLT_MAX;
		min_error_class = -1;
		for (m = 0; m < k; ++m) {
			if (min_error > NV_MAT_V(u, m, 0)) {
				min_error = NV_MAT_V(u, m, 0);
				min_error_class = m;
			}
		}

		if (max_error_class == min_error_class) {
			break;
		}
		/* クラスタ重心の更新 */
		nv_lbgu_update(means_tmp, data, labels_tmp, count_tmp,
			max_error_class, min_error_class, kmeans_max_epoch);

		if (nv_lbgu_progress_flag) {
			printf("%d: %ldms, error: %E\n", epoch, nv_clock() - t, error_best);
		}
	}

	nv_matrix_free(&e);
	nv_matrix_free(&u);
	nv_matrix_free(&means_tmp);
	nv_matrix_free(&count_tmp);
	nv_matrix_free(&labels_tmp);

	return epoch;
}

