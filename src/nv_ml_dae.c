/*
 * This file is part of libnv.
 *
 * Copyright (C) 2014 nagadomi@nurs.or.jp
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

#define NV_DAE_BATCH_SIZE 32
#define NV_DAE_BIAS 1.0f
#define NV_DAE_SPARSITY_BETA 0.25f
#define NV_DAE_WEIGHT_DECAY 0.0005f

static int nv_dae_progress_flag = 0;

void
nv_dae_progress(int onoff)
{
	nv_dae_progress_flag = onoff;
}

nv_dae_t *
nv_dae_alloc(int input, int hidden)
{
	nv_dae_t *dae = (nv_dae_t *)nv_malloc(sizeof(nv_dae_t));
	
	dae->input = input;
	dae->hidden = hidden;
	dae->noise = 0.0f;
	dae->sparsity = 0.0f;
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
	const float data_scale = 1.0f / data->m;
	const float input_norm_mean = sqrtf(0.8f * (dae->input_w->m + 1));
	float data_norm_mean;
	float input_scale;
	int j;
	
	data_norm_mean = 0.0f;
	for (j = 0; j < data->m; ++j) {
		data_norm_mean += nv_vector_norm(data, j) * data_scale;
	}
	input_scale = 1.0f / (data_norm_mean * input_norm_mean);
	
	nv_matrix_rand(dae->input_w, -0.5f * input_scale, 0.5f * input_scale);
	nv_matrix_zero(dae->input_bias);
	nv_matrix_zero(dae->hidden_bias);
}

void
nv_dae_noise(nv_dae_t *dae, float noise)
{
	dae->noise = noise;
}

void
nv_dae_sparsity(nv_dae_t *dae, float sparsity)
{
	dae->sparsity = sparsity;
}

static void
nv_dae_forward(const nv_dae_t *dae,
			   nv_dae_type_t type,
			   nv_matrix_t *input_y,
			   nv_matrix_t *output_y,
			   nv_matrix_t *corrupted_data, int batch_id)
{
	int j;
	for (j = 0; j < dae->input_w->m; ++j) {
		float y = NV_MAT_V(dae->input_bias, j, 0) * NV_DAE_BIAS;
		y += nv_vector_dot(corrupted_data, batch_id, dae->input_w, j);
		NV_MAT_V(input_y, batch_id, j) = NV_SIGMOID(y);
	}
	if (type == NV_DAE_SIGMOID) {
		for (j = 0; j < dae->input; ++j) {
			float y = NV_MAT_V(dae->hidden_bias, j, 0) * NV_DAE_BIAS;
			int i;
			for (i = 0; i < dae->hidden; ++i) {
				y += NV_MAT_V(input_y, batch_id, i) * NV_MAT_V(dae->input_w, i, j);
			}
			NV_MAT_V(output_y, batch_id, j) = NV_SIGMOID(y);
		}
	} else {
		for (j = 0; j < dae->input; ++j) {
			float y = NV_MAT_V(dae->hidden_bias, j, 0) * NV_DAE_BIAS;
			int i;
			for (i = 0; i < dae->hidden; ++i) {
				y += NV_MAT_V(input_y, batch_id, i) * NV_MAT_V(dae->input_w, i, j);
			}
			NV_MAT_V(output_y, batch_id, j) = y;
		}
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
	nv_dae_type_t type,
	const nv_matrix_t *output_y,
	const nv_matrix_t *input_y,
	const nv_matrix_t *activation_mean,
	const nv_matrix_t *corrupted_data,
	const nv_matrix_t *data,
	int *dj,
	const float lr)
{
	int i, j, batch_id;
	nv_matrix_t *output_bp = nv_matrix_alloc(dae->input, NV_DAE_BATCH_SIZE);
	nv_matrix_t *hidden_bp = nv_matrix_alloc(dae->input_w->m, NV_DAE_BATCH_SIZE);
	float weight_decay = dae->sparsity > 0.0f ? lr * NV_DAE_WEIGHT_DECAY : 0.0f;

	if (dae->sparsity > 0.0f) {
		/* sparse autoencoders */
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
		for (batch_id = 0; batch_id < NV_DAE_BATCH_SIZE; ++batch_id) {
			for (i = 0; i < output_bp->n; ++i) {
				float y_t = NV_MAT_V(output_y, batch_id, i) - NV_MAT_V(data, dj[batch_id], i);
				NV_MAT_V(output_bp, batch_id, i) = y_t;
			}
			for (i = 0; i < dae->input_w->m; ++i) {
				float y = nv_vector_dot(output_bp, batch_id, dae->input_w, i);
				float penalty =
					-dae->sparsity / (NV_MAT_V(activation_mean, 0, i) + FLT_EPSILON)
					+ (1.0f - dae->sparsity) / (1.0f - NV_MAT_V(activation_mean, 0, i) + FLT_EPSILON);
				NV_MAT_V(hidden_bp, batch_id, i) = 
					(y + NV_DAE_SPARSITY_BETA * penalty)
					 * (1.0f - NV_MAT_V(input_y, batch_id, i)) * NV_MAT_V(input_y, batch_id, i);
			}
		}
	} else {
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
		for (batch_id = 0; batch_id < NV_DAE_BATCH_SIZE; ++batch_id) {
			for (i = 0; i < output_bp->n; ++i) {
				float y_t = NV_MAT_V(output_y, batch_id, i) - NV_MAT_V(data, dj[batch_id], i);
				NV_MAT_V(output_bp, batch_id, i) = y_t;
			}
			for (i = 0; i < dae->input_w->m; ++i) {
				float y = nv_vector_dot(output_bp, batch_id, dae->input_w, i);
				NV_MAT_V(hidden_bp, batch_id, i) = 
					y * (1.0f - NV_MAT_V(input_y, batch_id, i)) * NV_MAT_V(input_y, batch_id, i);
			}
		}
	}
	
#ifdef _OPENMP
#pragma omp parallel for private(batch_id)
#endif
	for (i = 0; i < dae->input_w->n; ++i) {
		for (batch_id = 0; batch_id < NV_DAE_BATCH_SIZE; ++batch_id) {
			NV_MAT_V(dae->hidden_bias, i, 0) -= lr * NV_MAT_V(output_bp, batch_id, i) * NV_DAE_BIAS;
		}
	}
#ifdef _OPENMP
#pragma omp parallel for private(i, batch_id)
#endif
	for (j = 0; j < dae->input_w->m; ++j) {
		for (batch_id = 0; batch_id < NV_DAE_BATCH_SIZE; ++batch_id) {
			const float d1 = lr * NV_MAT_V(hidden_bp, batch_id, j);
			for (i = 0; i < dae->input_w->n; ++i) {
				const float d2 = lr * NV_MAT_V(output_bp, batch_id, i);
				NV_MAT_V(dae->input_w, j, i) -=
					(d1 * NV_MAT_V(corrupted_data, batch_id, i))
					+ (weight_decay * NV_MAT_V(dae->input_w, j, i))
					+ (d2 * NV_MAT_V(input_y, batch_id, j));
			}
			NV_MAT_V(dae->input_bias, j, 0) -= d1 * NV_DAE_BIAS;
		}
	}
	nv_matrix_free(&output_bp);
	nv_matrix_free(&hidden_bp);
}

void
nv_dae_corrupt(const nv_dae_t *dae,
			   nv_matrix_t *corrupted_data, int cj,
			   const nv_matrix_t *data, int dj)
{
	int i;
	for (i = 0; i < corrupted_data->n; ++i) {
		if (nv_rand() < dae->noise) {
			NV_MAT_V(corrupted_data, cj, i) = 0.0f;
		} else {
			NV_MAT_V(corrupted_data, cj, i) = NV_MAT_V(data, dj, i);
		}
	}
}

float
nv_dae_train_ex(nv_dae_t *dae,
				nv_dae_type_t type,
				const nv_matrix_t *data,
				float lr,
				int start_epoch, int end_epoch, int max_epoch)
{
	int i;
	int epoch = 1;
	float p;
	nv_matrix_t *input_y = nv_matrix_alloc(dae->hidden, NV_DAE_BATCH_SIZE);
	nv_matrix_t *output_y = nv_matrix_alloc(dae->input, NV_DAE_BATCH_SIZE);
	nv_matrix_t *corrupted_data = nv_matrix_alloc(dae->input, NV_DAE_BATCH_SIZE);
	nv_matrix_t *activation_mean = nv_matrix_alloc(dae->hidden, 2);
	nv_matrix_t *activation_tmp = nv_matrix_alloc(dae->hidden, 1);
	int activation_count;
	int *djs = nv_alloc_type(int, NV_DAE_BATCH_SIZE);
	int *rand_idx = nv_alloc_type(int, data->m);

	NV_ASSERT(data->m > NV_DAE_BATCH_SIZE);
	
	nv_matrix_zero(activation_mean);
	activation_count = 0;
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
			
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(+:correct, count, e)
#endif
			for (j = 0; j < NV_DAE_BATCH_SIZE; ++j) {
				int dj = rand_idx[i * NV_DAE_BATCH_SIZE + j];
				djs[j] = dj;
				nv_dae_corrupt(dae, corrupted_data, j, data, dj);
				nv_dae_forward(dae, type,
							   input_y, output_y,
							   corrupted_data, j);
				e += nv_dae_error(output_y, j, data, dj);
				count += 1;
			}
			for (j = 0; j < NV_DAE_BATCH_SIZE; ++j) {
				activation_count += 1;
				nv_vector_muls(activation_mean, 0,
							   activation_mean, 0,
							   (float)(activation_count - 1) / activation_count);
				nv_vector_muls(activation_tmp, 0, input_y, j, 1.0f / activation_count);
				nv_vector_add(activation_mean, 0, activation_mean, 0, activation_tmp, 0);
			}
			if (activation_count >= 1000) {
				nv_dae_backward(
					dae,
					type,
					output_y, input_y,
					activation_mean,
					corrupted_data,
					data, djs,
					lr);
			} // else wait for activation_mean
		}
		p = (float)correct / count;
		if (nv_dae_progress_flag) {
			printf("%d: E:%E, AM: %E, %ldms\n",
				   epoch, e / count / dae->input,
				   nv_vector_mean(activation_mean, 0),
				nv_clock() - tm);
			fflush(stdout);
		}
		activation_count = 1000;
	} while (epoch++ < end_epoch);
	nv_free(rand_idx);
	nv_free(djs);
	nv_matrix_free(&input_y);
	nv_matrix_free(&output_y);
	nv_matrix_free(&corrupted_data);
	nv_matrix_free(&activation_mean);
	nv_matrix_free(&activation_tmp);
	
	return p;
}

float
nv_dae_train(nv_dae_t *dae,
			 const nv_matrix_t *data,
			 float lr,
			 int start_epoch, int end_epoch, int max_epoch)
{
	return nv_dae_train_ex(dae, NV_DAE_SIGMOID,
						   data,
						   lr, start_epoch, end_epoch, max_epoch);
}

float
nv_dae_train_linear(nv_dae_t *dae,
					const nv_matrix_t *data,
					float lr, 
					int start_epoch, int end_epoch, int max_epoch)
{
	return nv_dae_train_ex(dae, NV_DAE_LINEAR,
						   data,
						   lr, start_epoch, end_epoch, max_epoch);
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
		NV_MAT_V(y, y_j, i) = NV_SIGMOID(z);
	}
}

/* conv utils */

// image  : 32x32
// patch  : 6
// patched: (32-6)x(32-6) = 26x26
// conved : 26x26
// pooling: size:3, stride:2, padding: 1
// pooled : 26/2 x 26/2 = 13x13
// patch  : 2
// patched: (13-2)x(13-2) = 11x11
// conved : 11x11
// pooling: size:3, stride:2, padding: 1
// pooled : 11/2 x 11/2 = 5x5
//

void
nv_dae_conv2d(const nv_dae_t *dae,
			  nv_matrix_t *output,
			  const nv_matrix_t *patches)
{
	int y;
	NV_ASSERT(output->n == dae->hidden);
	NV_ASSERT(patches->n == dae->input);
	NV_ASSERT(output->rows == patches->rows);
	NV_ASSERT(output->cols == patches->cols);

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (y = 0; y < patches->rows; ++y) {
		int x;
		for (x = 0; x < patches->cols; ++x) {
			nv_dae_encode(dae,
						  output, NV_MAT_M(output, y, x),
						  patches, NV_MAT_M(patches, y, x));
			
		}
	}
}


