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
#include "nv_ml_mlp.h"

/* 多層パーセプトロン
 * 2 Layer
 */

#define NV_MLP_MOMENTUM 0.9f
#define NV_MLP_WEIGHT_DECAY 0.0005f
#define NV_MLP_IR 0.001f
#define NV_MLP_HR 0.001f
#define NV_MLP_BATCH_SIZE 32
#define NV_MLP_BIAS (1.0f / NV_MLP_BATCH_SIZE)

#define nv_mlp_sigmoid(a) NV_SIGMOID(a)

static int nv_mlp_progress_flag = 0;

void nv_mlp_progress(int onoff)
{
	nv_mlp_progress_flag = onoff;
}


nv_mlp_t *nv_mlp_alloc(int input, int hidden, int k)
{
	nv_mlp_t *mlp = (nv_mlp_t *)nv_malloc(sizeof(nv_mlp_t));

	mlp->input = input;
	mlp->hidden = hidden;
	mlp->output = k;
	mlp->dropout = 0.0f;
	mlp->noise = 0.0f;
	mlp->input_w = nv_matrix_alloc(input, hidden);
	mlp->hidden_w = nv_matrix_alloc(mlp->input_w->m, k);
	mlp->input_bias = nv_matrix_alloc(1, hidden);
	mlp->hidden_bias = nv_matrix_alloc(1, k);

	return mlp;
}

void nv_mlp_dropout(nv_mlp_t *mlp, float dropout)
{
	mlp->dropout = dropout;
}

void
nv_mlp_noise(nv_mlp_t *mlp, float noise)
{
	mlp->noise = noise;
}

void nv_mlp_free(nv_mlp_t **mlp)
{
	if (*mlp) {
		nv_matrix_free(&(*mlp)->input_w);
		nv_matrix_free(&(*mlp)->input_bias);
		nv_matrix_free(&(*mlp)->hidden_w);
		nv_matrix_free(&(*mlp)->hidden_bias);
		nv_free(*mlp);
		*mlp = NULL;
	}
}

void nv_mlp_dump_c(FILE *out, const nv_mlp_t *mlp, const char *name, int static_variable)
{
	char var_name[4][1024];

	nv_snprintf(var_name[0], sizeof(var_name[0]), "%s_input_w", name);
	nv_matrix_dump_c(out, mlp->input_w, var_name[0], 1);
	nv_snprintf(var_name[1], sizeof(var_name[1]), "%s_hidden_w", name);
	nv_matrix_dump_c(out, mlp->hidden_w, var_name[1], 1);
	nv_snprintf(var_name[2], sizeof(var_name[2]), "%s_input_bias", name);
	nv_matrix_dump_c(out, mlp->input_bias, var_name[2], 1);
	nv_snprintf(var_name[3], sizeof(var_name[3]), "%s_hidden_bias", name);
	nv_matrix_dump_c(out, mlp->hidden_bias, var_name[3], 1);

	fprintf(out, "%snv_mlp_t %s = {\n %d, %d, %d, %f, %f, &%s, &%s, &%s, &%s\n};\n",
		static_variable ? "static ":"",
			name, mlp->input, mlp->hidden, mlp->output,
			mlp->dropout, mlp->noise,
		var_name[0],  var_name[1],  var_name[2],  var_name[3]);
	fflush(out);
}

static void
nv_mlp_softmax(nv_matrix_t *output_y, int oj,
			   const nv_matrix_t *hidden_y, int hj)
{
	float base = NV_MAT_V(hidden_y, hj, nv_vector_max_n(hidden_y, hj));
	float z = 0.0f;
	int n;
	for (n = 0; n < output_y->n; ++n) {
		NV_MAT_V(output_y, oj, n) = expf(NV_MAT_V(hidden_y, hj, n) - base);
		z += NV_MAT_V(output_y, oj, n);
	}
	nv_vector_divs(output_y, oj, output_y, oj, z);
}

/* クラス分類 */

int nv_mlp_predict_label(const nv_mlp_t *mlp, const nv_matrix_t *x, int xm)
{
	int m;
	int label = -1;
	float max_output = -FLT_MAX;
	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, 1);
	float dropout_scale = 1.0f - mlp->dropout;
	float noise_scale = 1.0f - mlp->noise;

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (m = 0; m < mlp->hidden; ++m) {
		float y = NV_MAT_V(mlp->input_bias, m, 0) * NV_MLP_BIAS;
		y += nv_vector_dot(x, xm, mlp->input_w, m) * noise_scale;
		NV_MAT_V(input_y, 0, m) = nv_mlp_sigmoid(y) * dropout_scale;
	}
	for (m = 0; m < mlp->output; ++m) {
		float y = NV_MAT_V(mlp->hidden_bias, m, 0) * NV_MLP_BIAS;
		y += nv_vector_dot(input_y, 0, mlp->hidden_w, m);
		if (max_output < y) {
			label = m;
			max_output = y;
		}
	}
	nv_matrix_free(&input_y);

	return label;
}

float nv_mlp_predict(const nv_mlp_t *mlp,
					 const nv_matrix_t *x, int xm, int cls)
{
	int m;
	float y;
	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, 1);
	nv_matrix_t *hidden_y = nv_matrix_alloc(mlp->output, 1);
	nv_matrix_t *output_y = nv_matrix_alloc(mlp->output, 1);
	float p;
	double dropout_scale = 1.0 - mlp->dropout;
	float noise_scale = 1.0f - mlp->noise;
	
#ifdef _OPENMP
#pragma omp parallel for private(y)
#endif
	for (m = 0; m < mlp->hidden; ++m) {
		y = NV_MAT_V(mlp->input_bias, m, 0) * NV_MLP_BIAS;
		y += nv_vector_dot(x, xm, mlp->input_w, m) * noise_scale;
		NV_MAT_V(input_y, 0, m) = nv_mlp_sigmoid(y) * dropout_scale;
	}

	for (m = 0; m < mlp->output; ++m) {
		y = NV_MAT_V(mlp->hidden_bias, m, 0) * NV_MLP_BIAS;
		y += nv_vector_dot(input_y, 0, mlp->hidden_w, m);
		NV_MAT_V(hidden_y, 0, m) = nv_mlp_sigmoid(y);
	}
	nv_mlp_softmax(output_y, 0, hidden_y, 0);
	p = NV_MAT_V(output_y, 0, cls);

	nv_matrix_free(&input_y);
	nv_matrix_free(&hidden_y);
	nv_matrix_free(&output_y);

	return p;
}

void
nv_mlp_predict_vector(const nv_mlp_t *mlp,
					  nv_matrix_t *p, int p_j,
					  const nv_matrix_t *x, int x_j)
{
	int m;
	float y;
	nv_matrix_t *input_y = nv_matrix_alloc(mlp->hidden, 1);
	nv_matrix_t *hidden_y = nv_matrix_alloc(mlp->output, 1);
	float dropout_scale = 1.0f - mlp->dropout;
	float noise_scale = 1.0f - mlp->noise;
	
#ifdef _OPENMP
#pragma omp parallel for private(y)
#endif
	for (m = 0; m < mlp->hidden; ++m) {
		y = NV_MAT_V(mlp->input_bias, m, 0) * NV_MLP_BIAS;
		y += nv_vector_dot(x, x_j, mlp->input_w, m) * noise_scale;
		NV_MAT_V(input_y, 0, m) = nv_mlp_sigmoid(y) * dropout_scale;
	}
	for (m = 0; m < mlp->output; ++m) {
		y = NV_MAT_V(mlp->hidden_bias, m, 0) * NV_MLP_BIAS;
		y += nv_vector_dot(input_y, 0, mlp->hidden_w, m);
		NV_MAT_V(hidden_y, 0, m) = nv_mlp_sigmoid(y);
	}
	nv_mlp_softmax(p, p_j, hidden_y, 0);

	nv_matrix_free(&input_y);
	nv_matrix_free(&hidden_y);
}


float nv_mlp_bagging_predict(const nv_mlp_t **mlp, int nmlp, 
							 const nv_matrix_t *x, int xm, int cls)
{
	float p = 0.0f;
	float factor = 1.0f / nmlp;
	int i;
	
	for (i = 0; i < nmlp; ++i) {
		p += factor * nv_mlp_predict(mlp[i], x, xm, cls);
	}

	return p;
}

/* 回帰 */
void nv_mlp_regression(const nv_mlp_t *mlp, 
					   const nv_matrix_t *x, int xm, nv_matrix_t *out, int om)
{
	int m;
	float y;
	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, 1);
	nv_matrix_t *hidden_y = nv_matrix_alloc(mlp->hidden_w->m, 1);

#ifdef _OPENMP
#pragma omp parallel for private(y)
#endif
	for (m = 0; m < mlp->input_w->m; ++m) {
		y = NV_MAT_V(mlp->input_bias, m, 0)  * NV_MLP_BIAS;
		y += nv_vector_dot(x, xm, mlp->input_w, m);
		NV_MAT_V(input_y, 0, m) = nv_mlp_sigmoid(y);
	}

	for (m = 0; m < mlp->hidden_w->m; ++m) {
		y = NV_MAT_V(mlp->hidden_bias, m, 0) * NV_MLP_BIAS;
		y += nv_vector_dot(input_y, 0, mlp->hidden_w, m);
		NV_MAT_V(hidden_y, 0, m) = y;
	}

	nv_vector_copy(out, om, hidden_y, 0);

	nv_matrix_free(&input_y);
	nv_matrix_free(&hidden_y);
}

/* 中間層ベクトル */
void nv_mlp_hidden_vector(const nv_mlp_t *mlp, 
						  const nv_matrix_t *x, int xm, nv_matrix_t *out, int om)
{
	int m;
	float y;
	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, 1);

	NV_ASSERT(input_y->n == out->n);

#ifdef _OPENMP
#pragma omp parallel for private(y)
#endif
	for (m = 0; m < mlp->input_w->m; ++m) {
		y = NV_MAT_V(mlp->input_bias, m, 0)  * NV_MLP_BIAS;
		y += nv_vector_dot(x, xm, mlp->input_w, m);
		NV_MAT_V(input_y, 0, m) = nv_mlp_sigmoid(y);
	}
	nv_vector_copy(out, om, input_y, 0);

	nv_matrix_free(&input_y);
}


/* training */

/* 初期化 */

/* グリッド上にランダムなガウス分布を作って初期化する.  */
void
nv_mlp_gaussian_init(nv_mlp_t *mlp, float var, int height, int width, int zdim)
{
	int m, n, x, y, z;
	nv_matrix_t *input_gaussian = nv_matrix_alloc(mlp->input, mlp->hidden);

	for (m = 0; m < mlp->input_w->m; ++m) {
		for (n = 0; n < mlp->input_w->n; ++n) {
			NV_MAT_V(mlp->input_w, m, n) = nv_rand() - 0.5f;
		}
		NV_MAT_V(mlp->input_bias, m, 0) = nv_rand() - 0.5f;
	}

	for (m = 0; m < input_gaussian->m; ++m) {
		float center_y = nv_rand();
		float center_x = nv_rand();

		for (y = 0; y < height; ++y) {
			float gy = (float)y / height - center_y;
			for (x = 0; x < width; ++x) {
				float gx = (float)x / width - center_x;
				float gaussian = expf(-(gy * gy) / var) * expf(-(gx * gx) / var);
				for (z = 0; z < zdim; ++z) {
					NV_MAT_V(input_gaussian, m, y * width * zdim + x * zdim + z) = gaussian;
				}
			}
		}
	}

	for (m = 0; m < mlp->input_w->m; ++m) {
		for (n = 0; n < mlp->input_w->n; ++n) {
			NV_MAT_V(mlp->input_w, m, n) *= NV_MAT_V(input_gaussian, m, n);
		}
	}

	for (m = 0; m < mlp->hidden_w->m; ++m) {
		for (n = 0; n < mlp->hidden_w->n; ++n) {
			NV_MAT_V(mlp->hidden_w, m, n) = nv_rand() - 0.5f;
		}
		NV_MAT_V(mlp->hidden_bias, m, 0) = nv_rand() - 0.5f;
	}
	nv_matrix_free(&input_gaussian);
}

/* 乱数で初期化  */
void
nv_mlp_init_rand(nv_mlp_t *mlp, const nv_matrix_t *data)
{
	const float data_scale = 1.0f / data->m;
	const float input_norm_mean = sqrtf(0.8f * (mlp->input_w->m + 1));
	const float hidden_norm_mean = sqrtf(0.8f * (mlp->hidden_w->m + 1));
	float data_norm_mean;
	float input_scale, hidden_scale;
	int j;
	
	data_norm_mean = 0.0f;
	for (j = 0; j < data->m; ++j) {
		data_norm_mean += nv_vector_norm(data, j) * data_scale;
	}
	input_scale = 1.0f / (data_norm_mean * input_norm_mean);
	hidden_scale = 1.0f / hidden_norm_mean;
	
	nv_matrix_rand(mlp->input_w, -0.5f * input_scale, 0.5f * input_scale);
	nv_matrix_rand(mlp->hidden_w, -0.5f * hidden_scale, 0.5f * hidden_scale);
	nv_matrix_zero(mlp->input_bias);
	nv_matrix_zero(mlp->hidden_bias);
}

void
nv_mlp_init(nv_mlp_t *mlp, const nv_matrix_t *data)
{
	nv_mlp_init_rand(mlp, data);
}

void
nv_mlp_make_t(nv_matrix_t *t, const nv_matrix_t *label)
{
	int m, n;

	NV_ASSERT(t->m == label->m);
	for (m = 0; m < t->m; ++m) {
		if (NV_MAT_VI(label, m, 0) == -1) {
			// nega
			for (n = 0; n < t->n; ++n) {
				NV_MAT_V(t, m, n) = 0.0f;
			}
		} else {
			for (n = 0; n < t->n; ++n) {
				NV_MAT_V(t, m, n) = NV_MAT_VI(label, m, 0) == n ? 1.0f:0.0f;
			}
		}
	}
}

/* クラス分類器の学習 */

float
nv_mlp_train(nv_mlp_t *mlp,
			 const nv_matrix_t *data, const nv_matrix_t *label,
			 int epoch)
{
	float ret;
	ret = nv_mlp_train_ex(mlp, data, label, NV_MLP_IR, NV_MLP_HR, 0, epoch, epoch);

	return ret;
}

float
nv_mlp_train_ex(nv_mlp_t *mlp,
				const nv_matrix_t *data,
				const nv_matrix_t *label,
				float ir, float hr,
				int start_epoch, int end_epoch,
				int max_epoch)
{
	float p;

	nv_matrix_t *t = nv_matrix_alloc(mlp->output, data->m);
	nv_mlp_make_t(t, label);
	p = nv_mlp_train_lex(mlp, data, label, t, ir, hr, start_epoch, end_epoch, max_epoch);
	nv_matrix_free(&t);

	return p;
}

static void
nv_mlp_train_accuracy(const nv_mlp_t *mlp,
					  const nv_matrix_t *data, const nv_matrix_t *label)
{
	int i;
	int output = mlp->output;
	int *correct_count = nv_alloc_type(int, output);
	int *error_count = nv_alloc_type(int, output);
	
	memset(correct_count, 0, sizeof(int) * output);
	memset(error_count, 0, sizeof(int) * output);
	for (i = 0; i < data->m; ++i) {
		int predict = nv_mlp_predict_label(mlp, data, i);
		int teach = (int)NV_MAT_V(label, i, 0);
		if (predict == teach) {
			++correct_count[teach];
		} else {
			++error_count[teach];
		}
	}
	for (i = 0; i < output; ++i) {
		if (correct_count[i] + error_count[i] > 0) {
			printf("%d: correct: %d, ng: %d, %f\n",
				   i, correct_count[i], error_count[i], 
				   (float)correct_count[i] / (float)(correct_count[i] + error_count[i]));
		} else {
			printf("%d: no data found\n", i);
		}
	}
	nv_free(correct_count);
	nv_free(error_count);
}

static void
nv_mlp_forward(const nv_mlp_t *mlp,
			   nv_matrix_t *input_y, int ij,
			   nv_matrix_t *hidden_y, int hj,
			   nv_matrix_t *corrupted_data, int cj)
{
	int m;
	for (m = 0; m < mlp->input_w->m; ++m) {
		if (nv_rand() > mlp->dropout) {
			float y = nv_vector_dot(corrupted_data, cj, mlp->input_w, m)
				+ NV_MAT_V(mlp->input_bias, m, 0) * NV_MLP_BIAS;
			NV_MAT_V(input_y, ij, m) = nv_mlp_sigmoid(y);
		} else {
			NV_MAT_V(input_y, ij, m) = 0.0f;
		}
	}
	for (m = 0; m < mlp->hidden_w->m; ++m) {
		float y = nv_vector_dot(input_y, ij, mlp->hidden_w, m)
			+ NV_MAT_V(mlp->hidden_bias, m, 0) * NV_MLP_BIAS;
		NV_MAT_V(hidden_y, hj, m) = y;
	}
}

static float
nv_mlp_error(const nv_matrix_t *output_y, int oj,
			 const nv_matrix_t *t, int dj)
{
	int n;
	float e = 0.0f;
	for (n = 0; n < output_y->n; ++n) {
		if (NV_MAT_V(output_y, oj, n) > FLT_EPSILON) {
			e += -NV_MAT_V(t, dj, n) * logf(NV_MAT_V(output_y, oj, n));
		}
	}
	return e;
}

static void
nv_mlp_backward(
	nv_mlp_t *mlp,
	nv_matrix_t *input_w_momentum,
	nv_matrix_t *input_bias_momentum,
	nv_matrix_t *hidden_w_momentum,
	nv_matrix_t *hidden_bias_momentum,
	const nv_matrix_t *output_y,
	const nv_matrix_t *input_y,
	const nv_matrix_t *corrupted_data,
	const nv_matrix_t *t,
	int *dj,
	const float ir,
	const float hr)
{
	int n, m, j;
	nv_matrix_t *output_bp = nv_matrix_alloc(mlp->output, NV_MLP_BATCH_SIZE);
	nv_matrix_t *hidden_bp = nv_matrix_alloc(mlp->input_w->m, NV_MLP_BATCH_SIZE);
	nv_matrix_t *input_w_grad = nv_matrix_alloc(mlp->input_w->n, mlp->input_w->m);
	nv_matrix_t *input_bias_grad = nv_matrix_alloc(mlp->input_bias->n,
													   mlp->input_bias->m);
	nv_matrix_t *hidden_w_grad = nv_matrix_alloc(mlp->hidden_w->n, mlp->hidden_w->m);
	nv_matrix_t *hidden_bias_grad = nv_matrix_alloc(mlp->hidden_bias->n,
													mlp->hidden_bias->m);
	nv_matrix_zero(input_w_grad);
	nv_matrix_zero(hidden_w_grad);
	nv_matrix_zero(input_bias_grad);
	nv_matrix_zero(hidden_bias_grad);

#ifdef _OPENMP
#pragma omp parallel for private(m, n)
#endif
	for (j = 0; j < NV_MLP_BATCH_SIZE; ++j) {
		for (n = 0; n < output_bp->n; ++n) {
			float y_t = NV_MAT_V(output_y, j, n) - NV_MAT_V(t, dj[j], n);
			float bp = y_t;
			NV_MAT_V(output_bp, j, n) = bp;
		}
		for (m = 0; m < mlp->hidden_w->n; ++m) {
			float y = 0.0f;
			for (n = 0; n < mlp->output; ++n) {
				y += NV_MAT_V(output_bp, j, n) * NV_MAT_V(mlp->hidden_w, n, m);
			}
			NV_MAT_V(hidden_bp, j, m) = 
				y * (1.0f - NV_MAT_V(input_y, j, m)) * NV_MAT_V(input_y, j, m);
		}
	}
#ifdef _OPENMP
#pragma omp parallel for private(m, j)
#endif
	for (n = 0; n < mlp->hidden_w->m; ++n) {
		for (j = 0; j < NV_MLP_BATCH_SIZE; ++j) {
			const float w = hr * NV_MAT_V(output_bp, j, n);
			for (m = 0; m < mlp->hidden_w->n; ++m) {
				NV_MAT_V(hidden_w_grad, n, m) += w * NV_MAT_V(input_y, j, m);
			}
			NV_MAT_V(hidden_bias_grad, n, 0) += w * NV_MLP_BIAS;
		}
	}
#ifdef _OPENMP
#pragma omp parallel for private(m, j)
#endif
	for (n = 0; n < mlp->input_w->m; ++n) {
		for (j = 0; j < NV_MLP_BATCH_SIZE; ++j) {
			const float w = ir * NV_MAT_V(hidden_bp, j, n);
			if (w != 0.0f) {
				for (m = 0; m < mlp->input_w->n; ++m) {
					NV_MAT_V(input_w_grad, n, m) += w * NV_MAT_V(corrupted_data, j, m);
				}
				NV_MAT_V(input_bias_grad, n, 0) += w * NV_MLP_BIAS;
			} // dropout
		}
	}

#ifdef _OPENMP
#pragma omp parallel for private(m)
#endif
	for (n = 0; n < mlp->hidden_w->m; ++n) {
		for (m = 0; m < mlp->hidden_w->n; ++m) {
			NV_MAT_V(hidden_w_momentum, n, m) =
				NV_MLP_MOMENTUM * NV_MAT_V(hidden_w_momentum, n, m)
				+ NV_MLP_WEIGHT_DECAY * hr * NV_MAT_V(mlp->hidden_w, n, m)
				+ NV_MAT_V(hidden_w_grad, n, m);
			NV_MAT_V(mlp->hidden_w, n, m) -= NV_MAT_V(hidden_w_momentum, n, m) * (1.0f - NV_MLP_MOMENTUM);
		}
		NV_MAT_V(hidden_bias_momentum, n, 0) =
			NV_MLP_MOMENTUM * NV_MAT_V(hidden_bias_momentum, n, 0)
			+ NV_MAT_V(hidden_bias_grad, n, 0);
		NV_MAT_V(mlp->hidden_bias, n, 0) -= NV_MAT_V(hidden_bias_momentum, n, 0) * (1.0f - NV_MLP_MOMENTUM);
	}
#ifdef _OPENMP
#pragma omp parallel for private(m)
#endif
	for (n = 0; n < mlp->input_w->m; ++n) {
		for (m = 0; m < mlp->input_w->n; ++m) {
			NV_MAT_V(input_w_momentum, n, m) =
				NV_MLP_MOMENTUM * NV_MAT_V(input_w_momentum, n, m)
				+ NV_MAT_V(input_w_grad, n, m);
			NV_MAT_V(mlp->input_w, n, m) -= NV_MAT_V(input_w_momentum, n, m) * (1.0f - NV_MLP_MOMENTUM);
		}
		NV_MAT_V(input_bias_momentum, n, 0) =
			NV_MLP_MOMENTUM * NV_MAT_V(input_bias_momentum, n, 0)
			+ NV_MAT_V(input_bias_grad, n, 0);
		NV_MAT_V(mlp->input_bias, n, 0) -= NV_MAT_V(input_bias_momentum, n, 0) * (1.0f - NV_MLP_MOMENTUM);
	}
	
	nv_matrix_free(&input_w_grad);
	nv_matrix_free(&hidden_w_grad);
	nv_matrix_free(&input_bias_grad);
	nv_matrix_free(&hidden_bias_grad);
	nv_matrix_free(&output_bp);
	nv_matrix_free(&hidden_bp);
}

static void
nv_mlp_corrupt(const nv_mlp_t *mlp,
			   nv_matrix_t *corrupted_data, int cj,
			   const nv_matrix_t *data, int dj)
{
	int i;
	if (mlp->noise > 0.0f) {
		for (i = 0; i < corrupted_data->n; ++i) {
			if (nv_rand() < mlp->noise) {
				NV_MAT_V(corrupted_data, cj, i) = 0.0f;
			} else {
				NV_MAT_V(corrupted_data, cj, i) = NV_MAT_V(data, dj, i);
			}
		}
	} else {
		nv_vector_copy(corrupted_data, cj, data, dj);
	}
}

float
nv_mlp_train_lex(nv_mlp_t *mlp,
				 const nv_matrix_t *data,
				 const nv_matrix_t *label,
				 const nv_matrix_t *t,
				 float ir, float hr, 
				 int start_epoch, int end_epoch, int max_epoch)
{
	int i;
	int epoch = 1;
	float p;
	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, NV_MLP_BATCH_SIZE);
	nv_matrix_t *hidden_y = nv_matrix_alloc(mlp->hidden_w->m, NV_MLP_BATCH_SIZE);
	nv_matrix_t *output_y = nv_matrix_alloc(mlp->output, NV_MLP_BATCH_SIZE);
	nv_matrix_t *corrupted_data = nv_matrix_alloc(mlp->input, NV_MLP_BATCH_SIZE);
	nv_matrix_t *input_w_momentum = nv_matrix_alloc(mlp->input_w->n, mlp->input_w->m);
	nv_matrix_t *input_bias_momentum = nv_matrix_alloc(mlp->input_bias->n,
													   mlp->input_bias->m);
	nv_matrix_t *hidden_w_momentum = nv_matrix_alloc(mlp->hidden_w->n, mlp->hidden_w->m);
	nv_matrix_t *hidden_bias_momentum = nv_matrix_alloc(mlp->hidden_bias->n,
														mlp->hidden_bias->m);
	
	int *djs = nv_alloc_type(int, NV_MLP_BATCH_SIZE);
	int *rand_idx = nv_alloc_type(int, data->m);
	
	NV_ASSERT(data->m > NV_MLP_BATCH_SIZE);

	nv_matrix_zero(input_w_momentum);
	nv_matrix_zero(hidden_w_momentum);
	nv_matrix_zero(input_bias_momentum);
	nv_matrix_zero(hidden_bias_momentum);

	epoch = start_epoch + 1;
	do {
		long tm;
		int correct = 0;
		float e = 0.0f;
		int count = 0;
		
		tm = nv_clock();
		nv_shuffle_index(rand_idx, 0, data->m);

		for (i = 0; i < data->m / NV_MLP_BATCH_SIZE; ++i) {
			int j;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(+:correct, count, e)
#endif
			for (j = 0; j < NV_MLP_BATCH_SIZE; ++j) {
				int label_correct;
				int dj = rand_idx[i * NV_MLP_BATCH_SIZE + j];
				djs[j] = dj;
				
				nv_mlp_corrupt(mlp, corrupted_data, j, data, dj);
				nv_mlp_forward(mlp, input_y, j, hidden_y, j,
							   corrupted_data, j);
				nv_mlp_softmax(output_y, j, hidden_y, j);
				e += nv_mlp_error(output_y, j, t, dj);
				label_correct = (int)NV_MAT_V(label, dj, 0);
				if (nv_vector_max_n(output_y, j) == label_correct) {
					++correct;
				}
				count += 1;
			}
			nv_mlp_backward(
				mlp,
				input_w_momentum, input_bias_momentum,
				hidden_w_momentum, hidden_bias_momentum,
				output_y, input_y, corrupted_data,
				t, djs,
				ir, hr);
		}
		p = (float)correct / count;
		if (nv_mlp_progress_flag) {
			printf("%d: E:%E, %f (%d/%d), %ldms\n",
				   epoch, e / count / mlp->output,
				   p, correct,
				   count, 
				nv_clock() - tm);
			if (nv_mlp_progress_flag >= 2) {
				nv_mlp_train_accuracy(mlp, data, label);
			}
			fflush(stdout);
		}
	} while (epoch++ < end_epoch);
	nv_free(rand_idx);
	nv_free(djs);
	nv_matrix_free(&input_y);
	nv_matrix_free(&hidden_y);
	nv_matrix_free(&output_y);
	nv_matrix_free(&corrupted_data);
	nv_matrix_free(&input_w_momentum);
	nv_matrix_free(&input_bias_momentum);
	nv_matrix_free(&hidden_w_momentum);
	nv_matrix_free(&hidden_bias_momentum);
	
	return p;
}

/* 回帰 */
void
nv_mlp_train_regression(
	nv_mlp_t *mlp,
	const nv_matrix_t *data,
	const nv_matrix_t *t,
	float ir, float hr,
	int start_epoch, int max_epoch)
{
	long tm;
	int m, n, im;
	int epoch = 1;
	float y, df, data_e;
	int *rand_index = (int *)nv_malloc(sizeof(int) * data->m);
	nv_matrix_t *rand_s = nv_matrix_alloc(mlp->input_w->m, 1);
	float rand_s_base = logf(3.0f + mlp->hidden) / 3.0f;
	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, 1);
	nv_matrix_t *hidden_y = nv_matrix_alloc(mlp->hidden_w->m, 1);
	nv_matrix_t *output_y = nv_matrix_alloc(mlp->output, 1);
	nv_matrix_t *output_bp = nv_matrix_alloc(mlp->output, 1);
	nv_matrix_t *hidden_bp = nv_matrix_alloc(mlp->input_w->m, 1);

	do {
		data_e = 0.0f;
		tm = nv_clock();

		nv_shuffle_index(rand_index, 0, data->m);
		for (im = 0; im < data->m; ++im) {
			float e;
			int dm = rand_index[im];

			for (m = 0; m < mlp->input_w->m; ++m) {
				NV_MAT_V(rand_s, 0, m) = (nv_rand() * rand_s_base) - (rand_s_base * 0.5f);
			}
#ifdef _OPENMP
#pragma omp parallel for private(y) //if (mlp->input * mlp->hidden > 10240)
#endif
			for (m = 0; m < mlp->input_w->m; ++m) {
				y = NV_MAT_V(mlp->input_bias, m, 0) * NV_MLP_BIAS;
				y += nv_vector_dot(data, dm, mlp->input_w, m);
				y = nv_mlp_sigmoid(y + NV_MAT_V(rand_s, 0, m)); // y + noise

				NV_MAT_V(input_y, 0, m) = y;
			}

#ifdef _OPENMP
#pragma omp parallel for private(y) if (mlp->output > 256)
#endif
			for (m = 0; m < mlp->hidden_w->m; ++m) {
				y = NV_MAT_V(mlp->hidden_bias, m, 0) * NV_MLP_BIAS;
				y += nv_vector_dot(input_y, 0, mlp->hidden_w, m);
				NV_MAT_V(hidden_y, 0, m) = y;
			}

			nv_vector_copy(output_y, 0, hidden_y, 0);

			e = 0.0f;
			for (n = 0; n < output_y->n; ++n) {
				float diff = NV_MAT_V(output_y, 0, n) - NV_MAT_V(t, dm, n);
				e += diff * diff;
			}
			data_e += e;

			for (n = 0; n < output_bp->n; ++n) {
				NV_MAT_V(output_bp, 0, n) = NV_MAT_V(output_y, 0, n) - NV_MAT_V(t, dm, n);
			}
			for (m = 0; m < mlp->hidden_w->n; ++m) {
				y = 0.0f;
				for (n = 0; n < mlp->output; ++n) {
					y += NV_MAT_V(output_bp, 0, n) * NV_MAT_V(mlp->hidden_w, n, m);
				}
				NV_MAT_V(hidden_bp, 0, m) = y * (1.0f - NV_MAT_V(input_y, 0, m)) * NV_MAT_V(input_y, 0, m);
			}
			/* I -= εΔI */
#ifdef _OPENMP
#pragma omp parallel for private(m, df) //if (mlp->input * mlp->hidden > 1024)
#endif
			for (n = 0; n < mlp->input_w->m; ++n) {
				for (m = 0; m < mlp->input_w->n; ++m) {
					df = ir * (NV_MAT_V(data, dm, m) * NV_MAT_V(hidden_bp, 0, n));
					NV_MAT_V(mlp->input_w, n, m) -= df;
				}
				df = ir * NV_MLP_BIAS * NV_MAT_V(hidden_bp, 0, n);
				NV_MAT_V(mlp->input_bias, n, 0) -= df;
			}

			/* H -= εΔH */
#ifdef _OPENMP
#pragma omp parallel for private(m, df) if (mlp->output > 256)
#endif
			for (n = 0; n < mlp->hidden_w->m; ++n) {
				for (m = 0; m < mlp->hidden_w->n; ++m) {
					df = hr * NV_MAT_V(input_y, 0, m) * NV_MAT_V(output_bp, 0, n);
					NV_MAT_V(mlp->hidden_w, n, m) -= df;
				}
				df = hr * NV_MLP_BIAS * NV_MAT_V(output_bp, 0, n);
				NV_MAT_V(mlp->hidden_bias, n, 0) -= df;
			}
		}
		if (nv_mlp_progress_flag) {
			printf("%d: %E, %E, %ldms\n", 
				epoch + start_epoch, data_e, sqrtf(data_e / data->m / mlp->output), nv_clock() - tm);
		}
	} while (epoch++ < max_epoch - start_epoch);

	nv_free(rand_index);
	nv_matrix_free(&input_y);
	nv_matrix_free(&rand_s);
	nv_matrix_free(&hidden_y);
	nv_matrix_free(&output_y);
	nv_matrix_free(&hidden_bp);
}
