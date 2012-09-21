/*
 * This file is part of libnv.
 *
 * Copyright (C) 2012 nagadomi@nurs.or.jp
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
#include "nv_ml_pa.h"

/* PA-II */

static int nv_pa_progress_flag = 0;
void nv_pa_progress(int onoff)
{
	nv_pa_progress_flag = onoff;
}

nv_pa_t *
nv_pa_alloc(int n, int k)
{
	nv_pa_t *pa = (nv_pa_t *)nv_alloc_type(nv_pa_t, 1);
	NV_ASSERT(k >= 2);
	
	pa->n = n;
	pa->k = k;
	if (pa->k == 2) {
		pa->w = nv_matrix_alloc(pa->n, 1);
	} else {
		pa->w = nv_matrix_alloc(pa->n, k);
	}
	
	return pa;
}

void
nv_pa_init(nv_pa_t *pa)
{
	nv_matrix_zero(pa->w);
}

void
nv_pa_free(nv_pa_t **pa)
{
	if (pa && *pa) {
		nv_matrix_free(&(*pa)->w);
		nv_free(*pa);
		*pa = NULL;
	}
}

static void
nv_pa_train_at(nv_pa_t *pa,
			   const nv_matrix_t *data,
			   const nv_matrix_t *labels,
			   int posi_label,
			   float c,
			   int max_epoch)
{
	int i, j, l;
	nv_matrix_t *norm = nv_matrix_alloc(1, data->m);
	for (i = 0; i < data->m; ++i) {
		NV_MAT_V(norm, i, 0) = nv_vector_norm(data, i);
	}
	for (l = 0; l < max_epoch; ++l) {
		float e = 0.0;
		long t = nv_clock();
		for (j = 0; j < data->m; ++j) {
			int rand_i = nv_rand_index(data->m);
			float y = posi_label == (int)NV_MAT_V(labels, rand_i, 0) ? 1.0f : -1.0f;
			float score = nv_vector_dot(pa->w, posi_label, data, rand_i);
			float l = 1.0f - score * y;
			if (l > 0.0) {
				/* float alpha = y * l / NV_MAT_V(norm, rand_i, 0); /// PA */
				/* float alpha = y * NV_MIN(c, l / NV_MAT_V(norm, rand_i, 0)); // PA-I */
				float alpha = y * l / (NV_MAT_V(norm, rand_i, 0) + 0.5f / c);// PA-II
				int i;
				for (i = 0; i < data->n; ++i) {
					NV_MAT_V(pa->w, posi_label, i) += alpha * NV_MAT_V(data, rand_i, i);
				}
			}
			e += (y * score < 0.0f) ? 1.0f : 0.0f;
		}
		if (nv_pa_progress_flag) {
			printf("nv_pa: label: %d, epoch: %d, ER: %f, %ldms\n",
				   posi_label, l, (float)e / data->m, nv_clock() - t);
		}
	}
	nv_matrix_free(&norm);
}

void
nv_pa_train(nv_pa_t *pa,
			const nv_matrix_t *data,
			const nv_matrix_t *labels,
			float c,
			int max_epoch)
{
	if (pa->k == 2) {
		nv_pa_train_at(pa, data, labels, 0, c, max_epoch);
	} else {
		int k;
#ifdef _OPENMP
#pragma omp	parallel for num_threads(nv_omp_procs())
#endif
		for (k = 0; k < pa->k; ++k) {
			nv_pa_train_at(pa, data, labels, k, c, max_epoch);
		}
	}
}

int
nv_pa_predict_label(const nv_pa_t *pa,
					nv_matrix_t *vec, int j)
{
	if (pa->n == 2) {
		return nv_vector_dot(pa->w, 0, vec, j) >= 0.0f ? 0 : 1;
	} else {
		int max_k = -1;
		float max_dot = -FLT_MAX;
		int k;
		for (k = 0; k < pa->k; ++k) {
			float dot = nv_vector_dot(pa->w, k, vec, j);
			if (dot > max_dot) {
				max_dot = dot;
				max_k = k;
			}
		}
		return max_k;
	}
}

void
nv_pa_dump_c(FILE *out,
			 const nv_pa_t *pa,
			 const char *name, int static_variable)
{
}
