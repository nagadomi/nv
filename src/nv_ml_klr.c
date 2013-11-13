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

#include "nv_ml.h"

static int nv_klr_progress_flag = 0;

void nv_klr_progress(int onoff)
{
	nv_klr_progress_flag = onoff;
}

void 
nv_klr_init(nv_lr_t *lr,         // k
			nv_matrix_t *count,  // k
			nv_matrix_t *labels, // data->m
			const nv_matrix_t *data,
			const nv_lr_param_t param)
{
	nv_matrix_t *means = nv_matrix_alloc(lr->n, lr->k);
	long t;

	NV_ASSERT(labels->m >= data->m);

	nv_matrix_zero(means);
	nv_matrix_zero(labels);
	nv_matrix_zero(count);

	if (nv_klr_progress_flag) {
		printf("nv_klr: 0: init++\n");
	}

	t = nv_clock();
	nv_kmeans(means, count, labels, data, lr->k, 50);
	//nv_lbgu(means, count, labels, data, lr->k, 5, 10);

	if (nv_klr_progress_flag) {
		printf("nv_klr: 0: init end: %ldms\n", nv_clock() - t);
	}
	nv_lr_init(lr, data);
	nv_lr_train(lr, data, labels, param);	
	nv_matrix_free(&means);

	if (nv_klr_progress_flag) {
		printf("nv_klr: 0: first step: %ldms\n", nv_clock() - t);
		fflush(stdout);
	}
}

int 
nv_klr_em(nv_lr_t *lr,         // k
		  nv_matrix_t *count,  // k
		  nv_matrix_t *labels, // data->m
		  const nv_matrix_t *data,
		  const nv_lr_param_t param,
		  const int max_epoch)
{
	int j, l;
	int processing = 1, last_processing = 0;
	int converge, epoch;
	long t;
	int relabel_count;
	int empty_class;
	float relabel_per;
	int num_threads = nv_omp_procs();
	nv_matrix_t *old_labels = nv_matrix_alloc(1, data->m);
	nv_matrix_t *count_tmp = nv_matrix_list_alloc(1, lr->k, num_threads);

	NV_ASSERT(labels->m >= data->m);
	NV_ASSERT(count->m >= lr->k);

	nv_matrix_copy(old_labels, 0, labels, 0, old_labels->m);

	epoch = 0;
	do {
		if (last_processing) {
			processing = 0;
		}
		t = nv_clock();
		nv_matrix_zero(count);
		nv_matrix_zero(count_tmp);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
		for (j = 0; j < data->m; ++j) {
			int label = nv_lr_predict_label(lr, data, j);
			int thread_idx = nv_omp_thread_id();

			NV_ASSERT(label < lr->k);

			NV_MAT_V(labels, j, 0) = (float)label;
			NV_MAT_LIST_V(count_tmp, thread_idx, label, 0) += 1.0f;
		}

		for (l = 0; l < num_threads; ++l) {
			for (j = 0; j < count->m; ++j) {
				NV_MAT_V(count, j, 0) += NV_MAT_LIST_V(count_tmp, l, j, 0);
			}
		}
		++epoch;

		/* 終了判定 */
		relabel_count = 0;
		for (j = 0; j < data->m; ++j) {
			if (NV_MAT_V(labels, j, 0) != NV_MAT_V(old_labels, j, 0)) {
				++relabel_count;
			}
		}
		empty_class = 0;
		for (j = 0; j < lr->k; ++j) {
			empty_class += (NV_MAT_V(count, j, 0) > 0.0f ? 0:1);
		}
		relabel_per = (float)relabel_count / data->m;
		if (epoch > 1) {
			converge = (relabel_per < 0.001f) ? 1:0;
		} else {
			converge =0;
		}

		if (nv_klr_progress_flag) {
			printf("nv_klr: %d: relabel: %f, empty_class: %d, %ldms\n",
			epoch, relabel_per, empty_class, nv_clock() -t);
			fflush(stdout);
		}
		t = nv_clock();

		if (converge) {
			/* 終了 */ 
			if (nv_klr_progress_flag) {
				printf("nv_klr: %d: finish:\n", epoch);
				fflush(stdout);				
			}
			processing = 0;
		} else {
			/* ラベル更新 */ 
			nv_matrix_copy(old_labels, 0, labels, 0, old_labels->m);

			/* LR再計算 */ 
			nv_lr_train(lr, data, labels, param);

			/* 最大試行回数判定 */ 
			if (max_epoch != 0
				&& epoch >= max_epoch)
			{
				/* 終了 */
				processing = 0;
			}
			if (nv_klr_progress_flag) {
				printf("nv_klr: %d: train: %ldms\n", epoch, nv_clock() -t);
				fflush(stdout);				
			}
		}
	} while (processing);

	nv_matrix_free(&old_labels);
	nv_matrix_free(&count_tmp);

	return converge;
}

void
nv_klr_train(nv_lr_t *lr,         // k
			 nv_matrix_t *count,  // k
			 nv_matrix_t *labels, // data->m
			 const nv_matrix_t *data,
			 const nv_lr_param_t param,
			 const int max_epoch)
{
	nv_klr_init(lr, count, labels, data,
				NV_LR_PARAM(10,
							param.grad_w,
							param.reg_type, param.reg_w, 1));
	nv_klr_em(lr, count, labels, data, param, max_epoch);
	nv_lr_train(lr, data, labels,
				NV_LR_PARAM(20,
							param.grad_w,
							param.reg_type, param.reg_w, 1));
}
