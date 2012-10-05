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

/* Normal Bayes */
/* 他と違ってtrainのあとにnv_nb_train_finishを呼ばないと学習が完了しないので注意 */


nv_nb_t *
nv_nb_alloc(int n, int k)
{
	int i;
	nv_nb_t *nb = (nv_nb_t *)nv_malloc(sizeof(nv_nb_t));

	nb->k = k;
	nb->n = n;
	nb->kcov = (nv_cov_t **)nv_malloc(sizeof(nv_cov_t *) * k);
	for (i = 0; i < k; ++i) {
		nb->kcov[i] = nv_cov_alloc(n);
	}
	nb->pk = nv_matrix_alloc(1, k);
	nv_matrix_zero(nb->pk);

	return nb;
}

void 
nv_nb_train(nv_nb_t *nb, const nv_matrix_t *data, int k)
{
	nv_cov_t *cov = nb->kcov[k];

	nv_cov_eigen(cov, data);
	NV_MAT_V(nb->pk, k, 0) = FLT_MAX;
}

void
nv_nb_train_all(nv_nb_t *nb,
				int k,
				const nv_matrix_t *data,
				const nv_matrix_t *labels)
{
	int i;

	NV_ASSERT(data->m == labels->m);
	
	for (i = 0; i < k; ++i) {
		int c = 0, j, l = 0;
		nv_matrix_t *tmp_data;
		
		for (j = 0; j < labels->m; ++j) {
			if ((int)NV_MAT_V(labels, j, 0) == i) {
				++c;
			}
		}
		NV_ASSERT(c > 0);
		tmp_data = nv_matrix_alloc(data->n, c);
		for (j = 0; j < labels->m; ++j) {
			if ((int)NV_MAT_V(labels, j, 0) == i) {
				nv_vector_copy(tmp_data, l++, data, j);
			}
		}
		nv_nb_train(nb, tmp_data, i);
		nv_matrix_free(&tmp_data);
	}
	nv_nb_train_finish(nb);
}

void 
nv_nb_train_finish(nv_nb_t *nb)
{
	float sum_frac;
	int k, sum = 0;

	for (k = 0; k < nb->k; ++k) {
		sum += nb->kcov[k]->data_m;
	}
	sum_frac = 1.0f / (float)sum;
	for (k = 0; k < nb->k; ++k) {
		NV_MAT_V(nb->pk, k, 0) = logf((float)nb->kcov[k]->data_m * sum_frac);
	}
}

int 
nv_nb_predict_label(const nv_nb_t *nb,
					const nv_matrix_t *x, int xm,
					int npca)
{
	static const float LOG_PI_0_5_NEGA = -0.5723649f;
	nv_matrix_t *y = nv_matrix_alloc(x->n, nb->k);
	int k, label = -1;
	float p_init;
	const int nb_k = nb->k;
	const int x_n = x->n;
	float mp = -FLT_MAX;

	if (npca == 0) {
		npca = (int)(x_n * 0.4f);
	}
	p_init = -LOG_PI_0_5_NEGA * npca;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
	for (k = 0; k < nb_k; ++k) {
		int n;
		float p = p_init + NV_MAT_V(nb->pk, k, 0);
		const nv_cov_t *cov = nb->kcov[k];

		NV_ASSERT(NV_MAT_V(nb->pk, k, 0) != FLT_MAX);

		nv_vector_sub(y, k, x, xm, nb->kcov[k]->u, 0);
		for (n = 0; n < npca; ++n) {
			float xv = nv_vector_dot(cov->eigen_vec, n, y, k);
			float ev = NV_MAT_V(cov->eigen_val, n, 0) * 2.0f;
			p += -(0.5f * logf(ev)) - (xv * xv) / (ev);
		}
#ifdef _OPENMP
#pragma omp critical (nv_nb_predict_label)
#endif
		{
			if (p > mp) {
				mp = p;
				label = k;
			}
		}
	}
	nv_matrix_free(&y);

	return label;
}

float 
nv_nb_predict(const nv_nb_t *nb,
			  const nv_matrix_t *x, int xm, int npca, int k)
{
	return 
		nv_gaussian_log_predict(npca, nb->kcov[k], x, xm) 
		+ NV_MAT_V(nb->pk, k, 0);
}


void 
nv_nb_free(nv_nb_t **nb)
{
	if (*nb) {
		int i;

		for (i = 0; i < (*nb)->k; ++i) {
			nv_cov_free(&(*nb)->kcov[i]);
		}

		nv_free((*nb)->kcov);
		nv_free((*nb)->pk);
		nv_free(*nb);
		*nb = NULL;
	}
}
