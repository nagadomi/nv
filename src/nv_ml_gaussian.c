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
#include "nv_num.h"
#include "nv_ml_gaussian.h"

/* ガウス分布 */


/* この関数は次元が高いと確率が小さくなりすぎて数値計算できないので
 * 次元が高い場合は, この関数値の対数であるnv_gaussian_log_predictを使うこと.
 */
float 
nv_gaussian_predict(const nv_cov_t *cov, const nv_matrix_t *x, int xm)
{
	int n;
	nv_matrix_t *y = nv_matrix_alloc(x->n, 2);
	float p = 1.0f;
	float d = (float)x->n;
	float delta2 = 0.0f;
	float lambda = 1.0f;

	nv_vector_sub(y, 0, x, xm, cov->u, 0);
	nv_matrix_mulv(y, 1, cov->eigen_vec, NV_MAT_TR, y, 0);
	for (n = 0; n < x->n; ++n) {
		float ev = NV_MAT_V(cov->eigen_val, n, 0);
		float xv = NV_MAT_V(y, 1, n);
		if (ev > 0.0f) {
			delta2 += (xv * xv) / ev;
			lambda *= sqrtf(ev);
		}
	}
	p = (1.0f / powf(2.0f * NV_PI, d / 2.0f)) * (1.0f / lambda) * expf(-0.5f * delta2);

	nv_matrix_free(&y);

	return p;
}

float 
nv_gaussian_log_predict(int npca, const nv_cov_t *cov, const nv_matrix_t *x, int xm)
{
	static const float LOG_PI_0_5_NEGA = -0.5723649f;
	nv_matrix_t *y = nv_matrix_alloc(x->n, 1);
	int n;
	float p;

	NV_ASSERT(npca <= x->n);

	if (npca == 0) {
		npca = (int)(x->n * 0.4f);
	}
	p = LOG_PI_0_5_NEGA * npca;

	nv_vector_sub(y, 0, x, xm, cov->u, 0);
	for (n = 0; n < npca; ++n) {
		float xv = nv_vector_dot(cov->eigen_vec, n, y, 0);
		float ev = NV_MAT_V(cov->eigen_val, n, 0) * 2.0f;
		p += -(0.5f * logf(ev)) - (xv * xv) / (ev);
	}
	nv_matrix_free(&y);

	return p;
}

float 
nv_gaussian_log_predict_range(int spca, int epca, const nv_cov_t *cov, const nv_matrix_t *x, int xm)
{
	static const float LOG_PI_0_5 = 0.5723649f;
	nv_matrix_t *y = nv_matrix_alloc(x->n, 1);
	float p = - (LOG_PI_0_5 * (epca - spca));
	int n;

	nv_vector_sub(y, 0, x, xm, cov->u, 0);
	for (n = spca; n < epca; ++n) {
		float xv = nv_vector_dot(cov->eigen_vec, n, y, 0);
		float ev = NV_MAT_V(cov->eigen_val, n, 0) * 2.0f;
		p += -(0.5f * logf(ev)) - (xv * xv) / (ev);
	}
	nv_matrix_free(&y);

	return p;
}
