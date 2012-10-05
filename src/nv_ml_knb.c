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

/* Normal/Naive Bayes clustering */

static int nv_knb_progress_flag = 0;
void nv_knb_progress(int onoff)
{
	nv_knb_progress_flag = onoff;
}

/* 初期化 */
/* k-means++での初期化後に最初の正規分布を推定する */
void 
nv_knb_init(nv_nb_t *nb,         // k
			nv_matrix_t *count,  // k
			nv_matrix_t *labels, // data->m
			const nv_matrix_t *data)
{
	nv_matrix_t *means = nv_matrix_alloc(nb->n, nb->k);
	int m, k;
	long t;

	NV_ASSERT(nb->n == data->n);
	NV_ASSERT(labels->m >= data->m);
	NV_ASSERT(count->m == nb->k);

	nv_matrix_zero(labels);
	nv_matrix_zero(count);

	if (nv_knb_progress_flag) {
		printf("knb: init++\n");
	}

	t = nv_clock();
	nv_kmeans_init_pp(means, nb->k, data, -1);

	if (nv_knb_progress_flag) {
		printf("knb: init end: %ldms\n", nv_clock() - t);
	}

	t = nv_clock();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
	for (m = 0; m < data->m; ++m) {
		int label = nv_nn(means, data, m);
#ifdef _OPENMP
#pragma omp critical (nv_knb_init)
#endif
		{
			/* ラベル決定 */
			NV_MAT_V(labels, m, 0) = (float)label;
			/* カウント */
			NV_MAT_V(count, label, 0) += 1.0f;
		}
	}
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
	for (k = 0; k < nb->k; ++k) {
		int c, m;// local
		nv_matrix_t *train_data = nv_matrix_alloc(nb->n, NV_MAT_VI(count, k, 0));

		for (c = m = 0; m < data->m; ++m) {
			if (NV_MAT_VI(labels, m, 0) == k) {
				nv_vector_copy(train_data, c++, data, m);
			}
		}
		NV_ASSERT(c == train_data->m);

		nv_nb_train(nb, train_data, k);
		nv_matrix_free(&train_data);
	}
	nv_nb_train_finish(nb);
	nv_matrix_free(&means);

	if (nv_knb_progress_flag) {
		printf("knb: first step: %ldms\n", nv_clock() - t);
	}
}


int 
nv_knb_em(nv_nb_t *nb,         // k
		  nv_matrix_t *count,  // k
		  nv_matrix_t *labels, // data->m
		  const nv_matrix_t *data,
		  const int npca,
		  const int max_epoch)
{
	int m;
	int processing = 1;
	int converge, epoch;
	long t;

	nv_matrix_t *old_labels = nv_matrix_alloc(labels->n, labels->m);

	NV_ASSERT(nb->n == data->n);
	NV_ASSERT(labels->m >= data->m);
	NV_ASSERT(count->m == nb->k);
	NV_ASSERT(npca <= nb->n);

	nv_matrix_copy(old_labels, 0, labels, 0, old_labels->m);

	epoch = 0;
	do {
		t = nv_clock();
		nv_matrix_zero(count);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
		for (m = 0; m < data->m; ++m) {
			int label = nv_nb_predict_label(nb, data, m, npca);

#ifdef _OPENMP
#pragma omp critical (nv_knb_em)
#endif
			{
				/* ラベル決定 */
				NV_MAT_V(labels, m, 0) = (float)label;
				/* カウント */
				NV_MAT_V(count, label, 0) += 1.0f;
			}
		}
		++epoch;

		/* 終了判定 */
		converge = 1;
		for (m = 0; m < data->m; ++m) {
			if (NV_MAT_V(labels, m, 0) != NV_MAT_V(old_labels, m, 0)) {
				converge = 0;
				break;
			}
		}

		if (converge) { 
			/* 終了 */
			processing = 0;
		} else {
			int i;
			/* ラベル更新 */
			nv_matrix_copy(old_labels, 0, labels, 0, old_labels->m);

			/* NB再計算 */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
			for (i = 0; i < nb->k; ++i) {
				int m, c; // local
				if (NV_MAT_V(count, i, 0) > 0.0f) {
					nv_matrix_t *train_data = nv_matrix_alloc(nb->n, NV_MAT_VI(count, i, 0));

					for (c = m = 0; m < data->m; ++m) {
						if (NV_MAT_VI(labels, m, 0) == i) {
							nv_vector_copy(train_data, c++, data, m);
						}
					}
					NV_ASSERT(c == train_data->m);

					nv_nb_train(nb, train_data, i);
					nv_matrix_free(&train_data);
				} else {
					nb->kcov[i]->data_m = 0;
				}
			}
			nv_nb_train_finish(nb);

			/* 最大試行回数判定 */
			if (max_epoch != 0
				&& epoch >= max_epoch)
			{
				/* 終了 */
				processing = 0;
			}
		}
		if (nv_knb_progress_flag) {
			printf("knb: em epoch: %d, %ldms\n", epoch, nv_clock() -t);
		}
	} while (processing);

	nv_matrix_free(&old_labels);

	return epoch;
}
