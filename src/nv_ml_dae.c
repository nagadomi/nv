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
#include "nv_ml_dae.h"

/**
 * Denoising Autoencoders
 */

#define NV_DAE_IR 0.001f
#define NV_DAE_HR 0.001f
#define NV_DAE_L2 0.0001f
#define NV_DAE_FOLOS_W 0.0001f
#define NV_DAE_BATCH_SIZE 20

#define nv_dae_sigmoid(a) NV_SIGMOID(a)
#define NV_DAE_BIAS 1.0f

static int nv_dae_progress_flag = 0;

void
nv_dae_progress(int onoff)
{
	nv_dae_progress_flag = onoff;
}

void
nv_dae_dropout(nv_dae_t *dae, float dropout)
{
	dae->dropout = dropout;
}

nv_dae_t *
nv_dae_alloc(int input, int hidden)
{
	nv_dae_t *dae = (nv_dae_t *)nv_malloc(sizeof(nv_dae_t));
	
	dae->input = input;
	dae->hidden = hidden;
	dae->dropout = 0.0f;
	dae->noise = 0.1f;
	dae->input_w = nv_matrix_alloc(input, hidden);
	dae->input_bias = nv_matrix_alloc(1, hidden);
	dae->hidden_bias = nv_matrix_alloc(1, input);
	
	return dae;
}

void
nv_dae_free(nv_dae_t **dae)
{
	if (*dae) {
		nv_matrix_free(&(*dae)->input_w);
		nv_matrix_free(&(*dae)->input_bias);
		nv_matrix_free(&(*dae)->hidden_bias);
		nv_free(*dae);
		*dae = NULL;
	}
}

void
nv_dae_init(nv_dae_t *dae, const nv_matrix_t *data)
{
	int j, i;
	float scale = sqrtf(6.0f / (dae->hidden + dae->input + 1.0f));

	nv_matrix_zero(dae->input_bias);
	nv_matrix_zero(dae->hidden_bias);
	for (j = 0; j < dae->input_w->m; ++j) {
		for (i = 0; i < dae->input_w->n; ++i) {
			NV_MAT_V(dae->input_w, j, i) = (nv_rand() - 0.5f) * scale;
		}
	}
}

void
nv_dae_noise(nv_dae_t *dae, float noise)
{
	dae->noise = noise;
}

static void
nv_dae_forward(nv_matrix_t *input_y, int ij,
			   nv_matrix_t *output_y, int oj,
			   nv_matrix_t *noise, int cj,
			   const nv_dae_t *dae,
			   const nv_matrix_t *hidden_w,
			   const nv_matrix_t *data, int dj)
{
	int m;
	for (m = 0; m < dae->input_w->m; ++m) {
		int i;
		if (nv_rand() > dae->dropout) {
			float y = NV_MAT_V(dae->input_bias, m, 0) * NV_DAE_BIAS;
			for (i = 0; i < data->n; ++i) {
				y += NV_MAT_V(noise, cj, i) * NV_MAT_V(data, dj, i) * NV_MAT_V(dae->input_w, m, i);
			}
			y = nv_dae_sigmoid(y);
			NV_MAT_V(input_y, ij, m) = y;
		} else {
			NV_MAT_V(input_y, ij, m) = 0.0f;
		}
	}
	for (m = 0; m < hidden_w->m; ++m) {
		float y = NV_MAT_V(dae->hidden_bias, m, 0) * NV_DAE_BIAS;
		y += nv_vector_dot(input_y, ij, hidden_w, m);
		NV_MAT_V(output_y, oj, m) = nv_dae_sigmoid(y);
	}
}

static float
nv_dae_error(const nv_matrix_t *output_y, int oj,
			 const nv_matrix_t *data, int dj)
{
	int i;
	float e = 0.0f;
	for (i = 0; i < output_y->n; ++i) {
		float z = NV_MAT_V(output_y, oj, i);
		float x = NV_MAT_V(data, dj, i);
		//e += x * logf(z) - (1.0f - x) * (1.0f - logf(z));
		e += ((x - z) * (x - z));
	}
	return e;
}

static void
nv_dae_backward(
	nv_dae_t *dae,
	const nv_matrix_t *output_y,
	const nv_matrix_t *input_y,
	const nv_matrix_t *hidden_w,
	const nv_matrix_t *noise,
	const nv_matrix_t *data,
	int *dj,
	const float ir,
	const float hr)
{
	int n, m, j;
	nv_matrix_t *output_bp = nv_matrix_alloc(dae->input, NV_DAE_BATCH_SIZE);
	nv_matrix_t *hidden_bp = nv_matrix_alloc(dae->input_w->m, NV_DAE_BATCH_SIZE);
	
#ifdef _OPENMP
#pragma omp parallel for private(m, n)
#endif
	for (j = 0; j < NV_DAE_BATCH_SIZE; ++j) {
		for (n = 0; n < output_bp->n; ++n) {
			float y_t = NV_MAT_V(output_y, j, n) - NV_MAT_V(data, dj[j], n);
			float bp = y_t;
			NV_MAT_V(output_bp, j, n) = bp;
		}
		for (m = 0; m < hidden_w->n; ++m) {
			float y = 0.0f;
			for (n = 0; n < dae->input; ++n) {
				y += NV_MAT_V(output_bp, j, n) * NV_MAT_V(hidden_w, n, m);
			}
			NV_MAT_V(hidden_bp, j, m) = 
				y * (1.0f - NV_MAT_V(input_y, j, m)) * NV_MAT_V(input_y, j, m);
		}
	}
#ifdef _OPENMP
#pragma omp parallel for private(m, j)
#endif
	for (n = 0; n < hidden_w->m; ++n) {
		for (j = 0; j < NV_DAE_BATCH_SIZE; ++j) {
			const float w = hr * NV_MAT_V(output_bp, j, n) * 0.5f;
			for (m = 0; m < hidden_w->n; ++m) {
				// tied: hidden_w = input_w'
				NV_MAT_V(dae->input_w, m, n) -= w * NV_MAT_V(input_y, j, m);
			}
			NV_MAT_V(dae->hidden_bias, n, 0) -= w * NV_DAE_BIAS;
		}
	}
#ifdef _OPENMP
#pragma omp parallel for private(m, j)
#endif
	for (n = 0; n < dae->input_w->m; ++n) {
		for (j = 0; j < NV_DAE_BATCH_SIZE; ++j) {
			const float w = ir * NV_MAT_V(hidden_bp, j, n) * 0.5f;
			if (w != 0.0f) {
				for (m = 0; m < dae->input_w->n; ++m) {
					NV_MAT_V(dae->input_w, n, m) -= NV_MAT_V(noise, j, m) * w * NV_MAT_V(data, dj[j], m);
				}
				NV_MAT_V(dae->input_bias, n, 0) -= w * NV_DAE_BIAS;
			} // else dropout
		}
	}
	nv_matrix_free(&output_bp);
	nv_matrix_free(&hidden_bp);
}

float
nv_dae_train(nv_dae_t *dae,
			 const nv_matrix_t *data,
			 float ir, float hr,
			 int start_epoch, int end_epoch, int max_epoch)
{
	int i;
	int epoch = 1;
	float p;
	nv_matrix_t *input_y = nv_matrix_alloc(dae->input_w->m, NV_DAE_BATCH_SIZE);
	nv_matrix_t *output_y = nv_matrix_alloc(dae->input, NV_DAE_BATCH_SIZE);
	nv_matrix_t *hidden_w = nv_matrix_alloc(dae->input_w->m, dae->input_w->n);
	nv_matrix_t *noise = nv_matrix_alloc(dae->input_w->n, NV_DAE_BATCH_SIZE);
	int *djs = nv_alloc_type(int, NV_DAE_BATCH_SIZE);
	int *rand_idx = nv_alloc_type(int, data->m);

	NV_ASSERT(data->m > NV_DAE_BATCH_SIZE);

	epoch = start_epoch + 1;
	do {
		long tm;
		int correct = 0;
		float e = 0.0f;
		int count = 0;

		tm = nv_clock();
		nv_shuffle_index(rand_idx, 0, data->m);

		for (i = 0; i < data->m / NV_DAE_BATCH_SIZE; ++i) {
			int j;
			
			nv_matrix_zero(noise);
			
			// tied: hidden_w = input_w'
			nv_matrix_tr_ex(hidden_w, dae->input_w);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(+:correct, count, e)
#endif
			for (j = 0; j < NV_DAE_BATCH_SIZE; ++j) {
				int dj = rand_idx[i * NV_DAE_BATCH_SIZE + j];
				int k;
				
				djs[j] = dj;
				for (k = 0; k < noise->n; ++k) {
					if (nv_rand() > dae->noise) {
						NV_MAT_V(noise, j, k) = 1.0f;
					}
				}
				nv_dae_forward(input_y, j, output_y, j,
							   noise, j,
							   dae, hidden_w, data, dj);
				e += nv_dae_error(output_y, j, data, dj);
				count += 1;
			}
			nv_dae_backward(
				dae,
				output_y, input_y, hidden_w,
				noise,
				data, djs,
				ir, hr);
		}
		p = (float)correct / count;
		if (nv_dae_progress_flag) {
			printf("%d: E:%E, %ldms\n",
				   epoch, e / count / dae->input,
				nv_clock() - tm);
			fflush(stdout);
		}
	} while (epoch++ < end_epoch);
	nv_free(rand_idx);
	nv_free(djs);
	nv_matrix_free(&hidden_w);
	nv_matrix_free(&input_y);
	nv_matrix_free(&output_y);
	nv_matrix_free(&noise);
	
	return p;
}

void
nv_dae_encode(const nv_dae_t *dae,
			  nv_matrix_t *y,
			  int y_j,
			  const nv_matrix_t *x,
			  int x_j)
{
	int i;

	NV_ASSERT(dae->input == x->n);
	NV_ASSERT(dae->hidden == y->n);

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (i = 0; i < dae->hidden; ++i) {
		float z = NV_MAT_V(dae->input_bias, i, 0)  * NV_DAE_BIAS;
		z += nv_vector_dot(x, x_j, dae->input_w, i);
		NV_MAT_V(y, y_j, i) = nv_dae_sigmoid(z);
	}
}
