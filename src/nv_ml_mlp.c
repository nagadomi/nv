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
#include "nv_ml_kmeans.h"
#include <stdlib.h>

/* 多層パーセプトロン
 * 2 Layer
 */

static int nv_mlp_progress_flag = 0;

void nv_mlp_progress(int onoff)
{
	nv_mlp_progress_flag = onoff;
}

#define nv_mlp_sigmoid(a) NV_SIGMOID(a)

double nv_mlp_sigmoid_d(double a)
{
	return 1.0 / (1.0 + exp(-a));
}

nv_mlp_t *nv_mlp_alloc(int input, int hidden, int k)
{
	nv_mlp_t *mlp = (nv_mlp_t *)nv_malloc(sizeof(nv_mlp_t));

	mlp->input = input;
	mlp->hidden = hidden;
	mlp->output = k;

	mlp->input_w = nv_matrix_alloc(input, hidden);
	mlp->hidden_w = nv_matrix_alloc(mlp->input_w->m, k);
	mlp->input_bias = nv_matrix_alloc(1, hidden);
	mlp->hidden_bias = nv_matrix_alloc(1, k);

	return mlp;
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

	fprintf(out, "%snv_mlp_t %s = {\n %d, %d, %d, &%s, &%s, &%s, &%s\n};\n",
		static_variable ? "static ":"",
		name, mlp->input, mlp->hidden, mlp->output,
		var_name[0],  var_name[1],  var_name[2],  var_name[3]);
	fflush(out);
}

/* クラス分類 */

int nv_mlp_predict_label(const nv_mlp_t *mlp, const nv_matrix_t *x, int xm)
{
	int m, n;
	int l = -1;
	float mp = -FLT_MAX, y;

	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, 1);
	nv_matrix_t *output_y = nv_matrix_alloc(mlp->output, 1);

#ifdef _OPENMP
#pragma omp parallel for private(y) 
#endif
	for (m = 0; m < mlp->hidden; ++m) {
		y = NV_MAT_V(mlp->input_bias, m, 0);
		y += nv_vector_dot(x, xm, mlp->input_w, m);
		NV_MAT_V(input_y, 0, m) = nv_mlp_sigmoid(y);
	}

	for (m = 0; m < mlp->output; ++m) {
		y = NV_MAT_V(mlp->hidden_bias, m, 0);
		y += nv_vector_dot(input_y, 0, mlp->hidden_w, m);
		NV_MAT_V(output_y, 0, m) = nv_mlp_sigmoid(y);
	}

	l = -1; // nega
	if (output_y->n == 1) {
		if (NV_MAT_V(output_y, 0, 0) > 0.5f) {
			l = 0;
		} else {
			l = 1;
		}
	} else {
		for (n = 0; n < output_y->n; ++n) {
			if (mp < NV_MAT_V(output_y, 0, n)
//				&& NV_MAT_V(output_y, 0, n) > 0.5f
				) 
			{
				mp = NV_MAT_V(output_y, 0, n);
				l = n;
			}
		}
	}

	nv_matrix_free(&input_y);
	nv_matrix_free(&output_y);

	return l;
}

double nv_mlp_predict_d(const nv_mlp_t *mlp,
						const nv_matrix_t *x, int xm, int cls)
{
	int m;
	float y;
	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, 1);
	nv_matrix_t *output_z = nv_matrix_alloc(mlp->output, 1);
	nv_matrix_t *output_y = nv_matrix_alloc(mlp->output, 1);
	double p;

#ifdef _OPENMP
#pragma omp parallel for private(y)
#endif
	for (m = 0; m < mlp->hidden; ++m) {
		y = NV_MAT_V(mlp->input_bias, m, 0);
		y += nv_vector_dot(x, xm, mlp->input_w, m);
		NV_MAT_V(input_y, 0, m) = nv_mlp_sigmoid(y);
	}

	for (m = 0; m < mlp->output; ++m) {
		y = NV_MAT_V(mlp->hidden_bias, m, 0);;
		y += nv_vector_dot(input_y, 0, mlp->hidden_w, m);
		NV_MAT_V(output_z, 0, m) = y;
		NV_MAT_V(output_y, 0, m) = y;
	}
	p = (double)NV_MAT_V(output_z, 0, cls);
	p = nv_mlp_sigmoid_d(p);

	nv_matrix_free(&input_y);
	nv_matrix_free(&output_y);
	nv_matrix_free(&output_z);

	return p;
}


float nv_mlp_predict(const nv_mlp_t *mlp,
					 const nv_matrix_t *x, int xm, int cls)
{
	int m;
	float y;
	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, 1);
	nv_matrix_t *output_y = nv_matrix_alloc(mlp->output, 1);
	float p;

#ifdef _OPENMP
#pragma omp parallel for private(y)
#endif
	for (m = 0; m < mlp->hidden; ++m) {
		y = NV_MAT_V(mlp->input_bias, m, 0);
		y += nv_vector_dot(x, xm, mlp->input_w, m);
		NV_MAT_V(input_y, 0, m) = nv_mlp_sigmoid(y);
	}

	for (m = 0; m < mlp->output; ++m) {
		y = NV_MAT_V(mlp->hidden_bias, m, 0);;
		y += nv_vector_dot(input_y, 0, mlp->hidden_w, m);
		NV_MAT_V(output_y, 0, m) = nv_mlp_sigmoid(y);
	}
	p = NV_MAT_V(output_y, 0, cls);

	nv_matrix_free(&input_y);
	nv_matrix_free(&output_y);

	return p;
}

double nv_mlp_bagging_predict_d(const nv_mlp_t **mlp, int nmlp, 
							   const nv_matrix_t *x, int xm, int cls)
{
	double p = 0.0f;
	double factor = 1.0 / nmlp;
	int i;
	
	for (i = 0; i < nmlp; ++i) {
		p += factor * nv_mlp_predict_d(mlp[i], x, xm, cls);
	}

	return p;
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
		y = NV_MAT_V(mlp->input_bias, m, 0);
		y += nv_vector_dot(x, xm, mlp->input_w, m);
		NV_MAT_V(input_y, 0, m) = nv_mlp_sigmoid(y);
	}

	for (m = 0; m < mlp->hidden_w->m; ++m) {
		y = NV_MAT_V(mlp->hidden_bias, m, 0);
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
		y = NV_MAT_V(mlp->input_bias, m, 0);
		y += nv_vector_dot(x, xm, mlp->input_w, m);
		NV_MAT_V(input_y, 0, m) = nv_mlp_sigmoid(y);
	}
	nv_vector_copy(out, om, input_y, 0);

	nv_matrix_free(&input_y);
}


/* training */

#define NV_MLP_MAX_EPOCH 2000
#define NV_MLP_IR 0.01f
#define NV_MLP_HR 0.0001f
#define NV_MLP_L2 0.0001f
#define NV_MLP_FOLOS_W 0.0001f

/* 初期化 */

/* グリッド上にランダムなガウス分布を作って初期化する. これが一番いい.  */
void nv_mlp_gaussian_init(nv_mlp_t *mlp, float var, int height, int width, int zdim)
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

void 
nv_mlp_parts_init(nv_mlp_t *mlp, int r, int height, int width, int zdim)
{
	int m, n, x, y, z;

	for (m = 0; m < mlp->input_w->m; ++m) {
		float center_y = nv_rand() * (height - 1);
		float center_x = nv_rand() * (width - 1);

		for (y = 0; y < height; ++y) {
			for (x = 0; x < width; ++x) {
				if (sqrtf((center_y - y) * (center_y - y) + (center_x - x) * (center_x - x)) > r) {
					for (z = 0; z < zdim; ++z) {
						NV_MAT_V(mlp->input_w, m, y * width * zdim + x * zdim + z) = 0.0f;
					}
				} else {
					for (z = 0; z < zdim; ++z) {
						NV_MAT_V(mlp->input_w, m, y * width * zdim + x * zdim + z) = nv_rand() - 0.5f;
					}
				}
			}
		}
	}
	for (m = 0; m < mlp->input_w->m; ++m) {
		NV_MAT_V(mlp->input_bias, m, 0) = nv_rand() - 0.5f;
	}
	for (m = 0; m < mlp->hidden_w->m; ++m) {
		for (n = 0; n < mlp->hidden_w->n; ++n) {
			NV_MAT_V(mlp->hidden_w, m, n) = nv_rand() - 0.5f;
		}
		NV_MAT_V(mlp->hidden_bias, m, 0) = nv_rand() - 0.5f;
	}
}


void nv_mlp_init_nonnegative(nv_mlp_t *mlp)
{
	int m, n;

	for (m = 0; m < mlp->input_w->m; ++m) {
		for (n = 0; n < mlp->input_w->n; ++n) {
			NV_MAT_V(mlp->input_w, m, n) = nv_rand();
		}
		nv_vector_normalize_L2(mlp->input_w, m);
		NV_MAT_V(mlp->input_bias, m, 0) = -nv_rand();
	}

	for (m = 0; m < mlp->hidden_w->m; ++m) {
		for (n = 0; n < mlp->hidden_w->n; ++n) {
			NV_MAT_V(mlp->hidden_w, m, n) = nv_rand();
		}
		nv_vector_normalize_L2(mlp->hidden_w, m);
		NV_MAT_V(mlp->hidden_bias, m, 0) = -nv_rand();
	}
}


void nv_mlp_init_rand(nv_mlp_t *mlp)
{
	int m, n;
	
	for (m = 0; m < mlp->input_w->m; ++m) {
		for (n = 0; n < mlp->input_w->n; ++n) {
			NV_MAT_V(mlp->input_w, m, n) = nv_rand() - 0.5f;
		}
		NV_MAT_V(mlp->input_bias, m, 0) = nv_rand() - 0.5f;
	}

	for (m = 0; m < mlp->hidden_w->m; ++m) {
		for (n = 0; n < mlp->hidden_w->n; ++n) {
			NV_MAT_V(mlp->hidden_w, m, n) = nv_rand() - 0.5f;
		}
		NV_MAT_V(mlp->hidden_bias, m, 0) = nv_rand() - 0.5f;
	}
}

void nv_mlp_init_kmeans(nv_mlp_t *mlp, const nv_matrix_t *data)
{
	int m, n;
	nv_matrix_t *means = nv_matrix_alloc(mlp->input, mlp->hidden);
	nv_matrix_t *labels = nv_matrix_alloc(1, data->m);
	nv_matrix_t *count = nv_matrix_alloc(1, mlp->hidden);
	
	nv_kmeans(means, count, labels, data, mlp->hidden, 50);
	
	for (m = 0; m < mlp->input_w->m; ++m) {
		nv_vector_copy(mlp->input_w, m, means, m);
		nv_vector_normalize_L2(mlp->input_w, m);
		NV_MAT_V(mlp->input_bias, m, 0) = nv_rand() - 0.5f;
	}
	for (m = 0; m < mlp->hidden_w->m; ++m) {
		for (n = 0; n < mlp->hidden_w->n; ++n) {
			NV_MAT_V(mlp->hidden_w, m, n) = nv_rand() - 0.5f;
		}
		NV_MAT_V(mlp->hidden_bias, m, 0) = nv_rand() - 0.5f;
	}
	nv_matrix_free(&means);
	nv_matrix_free(&labels);
	nv_matrix_free(&count);
}

void nv_mlp_init(nv_mlp_t *mlp)
{
	nv_mlp_init_rand(mlp);
}

void nv_mlp_make_t(nv_matrix_t *t, const nv_matrix_t *label)
{
	int m, n;

	NV_ASSERT(t->m == label->m);
	for (m = 0; m < t->m; ++m) {
		if (NV_MAT_V(label, m, 0) == -1.0f) {
			// nega
			for (n = 0; n < t->n; ++n) {
				NV_MAT_V(t, m, n) = 0.0f;
			}
		} else {
			for (n = 0; n < t->n; ++n) {
				NV_MAT_V(t, m, n) = NV_MAT_V(label, m, 0) == (float)n ? 1.0f:0.0f;
			}
		}
	}
}

/* クラス分類器の学習 */

float nv_mlp_train(nv_mlp_t *mlp, const nv_matrix_t *data, const nv_matrix_t *label, int epoch)
{
	nv_matrix_t *ir = nv_matrix_alloc(1, mlp->output == 1 ? 2 : mlp->output);
	nv_matrix_t *hr = nv_matrix_alloc(1, mlp->output == 1 ? 2 : mlp->output);
	int j;
	float ret;
	for (j = 0; j < mlp->output; ++j) {
		NV_MAT_V(ir, j, 0) = NV_MLP_IR;
		NV_MAT_V(hr, j, 0) = NV_MLP_HR;
	}
	ret = nv_mlp_train_ex(mlp, data, label, ir, hr, 1, 0, epoch, epoch);

	nv_matrix_free(&ir);
	nv_matrix_free(&hr);

	return ret;
}

float nv_mlp_train_ex(nv_mlp_t *mlp,
					  const nv_matrix_t *data,
					  const nv_matrix_t *label,
					  const nv_matrix_t *ir, const nv_matrix_t *hr,
					  int train_thresh,
					  int start_epoch, int end_epoch,
					  int max_epoch)
{
	float p;

	nv_matrix_t *t = nv_matrix_alloc(mlp->output, data->m);
	nv_mlp_make_t(t, label);
	p = nv_mlp_train_lex(mlp, data, label, t, ir, hr,train_thresh, start_epoch, end_epoch, max_epoch);
	nv_matrix_free(&t);

	return p;
}

float nv_mlp_train_lex(nv_mlp_t *mlp,
					 const nv_matrix_t *data,
					 const nv_matrix_t *label,
					 const nv_matrix_t *t,
					 const nv_matrix_t *ir, const nv_matrix_t *hr, 
					 int train_thresh,
					 int start_epoch, int end_epoch, int max_epoch)
{
	long tm;
	int m, n, im, ok;
	int epoch = 1;
	float p;
	float y, data_e, bp_sum;
	int *rand_idx = (int *)nv_malloc(sizeof(int) * data->m);
	int label_ok;
	int do_train;
	
	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, 1);
	nv_matrix_t *hidden_y = nv_matrix_alloc(mlp->hidden_w->m, 1);
	nv_matrix_t *output_y = nv_matrix_alloc(mlp->output, 1);
	nv_matrix_t *output_bp = nv_matrix_alloc(mlp->output, 1);
	nv_matrix_t *hidden_bp = nv_matrix_alloc(mlp->input_w->m, 1);
	nv_matrix_t *rand_s = nv_matrix_alloc(mlp->input_w->m, 1);
	float rand_s_base = logf(mlp->hidden) / logf(32.0);
	
	epoch = start_epoch + 1;
	do {
		ok = 0;
		data_e = 0.0f;
		tm = nv_clock();
		nv_shuffle_index(rand_idx, 0, data->m);
		for (im = 0; im < data->m; ++im) {
			float e;
			float mp = -FLT_MAX;
			int l = -1;
			int dm = rand_idx[im];

			for (m = 0; m < mlp->input_w->m; ++m) {
				NV_MAT_V(rand_s, 0, m) = (nv_rand() * rand_s_base) - (rand_s_base * 0.5f);
			}
#ifdef _OPENMP
#pragma omp parallel for private(y) //if (mlp->input * mlp->hidden > 10240)
#endif
			for (m = 0; m < mlp->input_w->m; ++m) {
				y = NV_MAT_V(mlp->input_bias, m, 0);
				y += nv_vector_dot(data, dm, mlp->input_w, m);
				y = nv_mlp_sigmoid(y + NV_MAT_V(rand_s, 0, m)); // y + noise

				NV_MAT_V(input_y, 0, m) = y;
			}

#ifdef _OPENMP
#pragma omp parallel for private(y) if (mlp->output > 256)
#endif
			for (m = 0; m < mlp->hidden_w->m; ++m) {
				y = NV_MAT_V(mlp->hidden_bias, m, 0);
				y += nv_vector_dot(input_y, 0, mlp->hidden_w, m);
				NV_MAT_V(hidden_y, 0, m) = y;
			}

			for (m = 0; m < mlp->hidden_w->m; ++m) {
				NV_MAT_V(output_y, 0, m) = nv_mlp_sigmoid(NV_MAT_V(hidden_y, 0, m));
			}

			do_train = 0;
			e = 0.0f;
			for (n = 0; n < output_y->n; ++n) {
				float y = NV_MAT_V(output_y, 0, n);
				float yt = NV_MAT_V(t, dm, n);
				if (y == 0.0f) {
					e += (yt * 0.0f + (1.0f - yt) * logf(1.0f - y)) * -1.0f;
				} else if (y == 1.0f) {
					e += (yt * logf(y) + (1.0f - yt) * 0.0f) * - 1.0f;
				} else {
					e += (yt * logf(y) + (1.0f - yt) * logf(1.0f - y)) * -1.0f;
				}
			}
			data_e += e;
			mp = -FLT_MAX;
			l = -1; // nega
			label_ok = (int)NV_MAT_V(label, dm, 0);

			if (output_y->n == 1) {
				if (NV_MAT_V(output_y, 0, 0) > 0.5f) {
					l = 0;
				} else {
					l = 1;
				}
				if (label_ok == 0) {
					if (NV_MAT_V(output_y, 0, 0) < 0.995f) {
						do_train = 1;
					}
				} else {
					if (NV_MAT_V(output_y, 0, 0) > 0.005f) {
						do_train = 1;
					}
				}
			} else {
				for (n = 0; n < output_y->n; ++n) {
					if (//NV_MAT_V(output_y, 0, n) > 0.5f
						//&&
						mp < NV_MAT_V(output_y, 0, n)) 
					{
						mp = NV_MAT_V(output_y, 0, n);
						l = n;
					}
					if (n == label_ok) {
						if (NV_MAT_V(output_y, 0, n) < 0.995f) {
							do_train = 1;
						}
					} else {
						if (NV_MAT_V(output_y, 0, n) > 0.005f) {
							do_train = 1;
						}
					}
				}
			}
			if (l == label_ok) {
				++ok;
			}
			if (train_thresh == 0 || do_train) {
				bp_sum = 0.0f;
				for (n = 0; n < output_bp->n; ++n) {
					float y = NV_MAT_V(hidden_y, 0, n);
					float yt = NV_MAT_V(t, dm, n);
					float expy = expf(y);
					float bp = -((2.0f * yt - 1.0f) * expy + yt) 
						/ (expf(2.0f * y) + 2.0f * expy + 1.0f);
					bp_sum += bp;
					NV_MAT_V(output_bp, 0, n) = bp;
				}
				if (bp_sum != 0.0f) {
					for (m = 0; m < mlp->hidden_w->n; ++m) {
						y = 0.0f;
						for (n = 0; n < mlp->output; ++n) {
							y += NV_MAT_V(output_bp, 0, n) * NV_MAT_V(mlp->hidden_w, n, m);
						}
						NV_MAT_V(hidden_bp, 0, m) = 
							y * (1.0f - NV_MAT_V(input_y, 0, m)) * NV_MAT_V(input_y, 0, m);
					}
#ifdef _OPENMP
#pragma omp parallel for private(m) // if (mlp->input * mlp->hidden > 10240)
#endif
					for (n = 0; n < mlp->input_w->m; ++n) {
						for (m = 0; m < mlp->input_w->n; ++m) {
							NV_MAT_V(mlp->input_w, n, m) -= 
								NV_MAT_V(ir, label_ok, 0) 
								* NV_MAT_V(data, dm, m) * NV_MAT_V(hidden_bp, 0, n)
								;
						}
						NV_MAT_V(mlp->input_bias, n, 0) -=
							NV_MAT_V(ir, label_ok, 0) * 1.0f * NV_MAT_V(hidden_bp, 0, n)
							;
					}
#ifdef _OPENMP
#pragma omp parallel for private(m) if (mlp->output > 256)
#endif
					for (n = 0; n < mlp->hidden_w->m; ++n) {
						for (m = 0; m < mlp->hidden_w->n; ++m) {
							NV_MAT_V(mlp->hidden_w, n, m) -= 
								NV_MAT_V(hr, label_ok, 0) 
								* NV_MAT_V(input_y, 0, m) * NV_MAT_V(output_bp, 0, n)
								;
						}
						NV_MAT_V(mlp->hidden_bias, n, 0) -=
							NV_MAT_V(hr, label_ok, 0) * 1.0f * NV_MAT_V(output_bp, 0, n)
							;
					}
				}
			}
		}
		p = (float)ok / data->m;
		if (nv_mlp_progress_flag) {
			printf("%d: E:%E, %f (%d/%d), %ldms\n",
				epoch, data_e / data->m / mlp->output,
				p, ok,
				data->m, 
				nv_clock() - tm);
			if (nv_mlp_progress_flag >= 2) {
				int i;
				int output = mlp->output > 1 ? mlp->output : 2;
				int *ok_count = nv_alloc_type(int, output);
				int *ng_count = nv_alloc_type(int, output);

				memset(ok_count, 0, sizeof(int) * output);
				memset(ng_count, 0, sizeof(int) * output);
				for (i = 0; i < data->m; ++i) {
					int predict = nv_mlp_predict_label(mlp, data, i);
					int teach = (int)NV_MAT_V(label, i, 0);
					if (predict == teach) {
						++ok_count[teach];
					} else {
						++ng_count[teach];
					}
				}
				for (i = 0; i < output; ++i) {
					if (ok_count[i] + ng_count[i] > 0) {
						printf("%d: ok: %d, ng: %d, %f\n",
							i, ok_count[i], ng_count[i], 
							(float)ok_count[i] / (float)(ok_count[i] + ng_count[i]));
					} else {
						printf("%d: no data found\n", i);
					}
				}
				nv_free(ok_count);
				nv_free(ng_count);
			}
			fflush(stdout);
		}
	} while (epoch++ < end_epoch);

	nv_matrix_free(&input_y);
	nv_matrix_free(&hidden_y);
	nv_matrix_free(&output_y);
	nv_matrix_free(&hidden_bp);
	nv_matrix_free(&output_bp);
	nv_matrix_free(&rand_s);
	nv_free(rand_idx);

	return p;
}

/* 回帰 */
void nv_mlp_train_regression(
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
	int *rand_idx = (int *)nv_malloc(sizeof(int) * data->m);
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

		nv_shuffle_index(rand_idx, 0, data->m);
		for (im = 0; im < data->m; ++im) {
			float e;
			int dm = rand_idx[im];

			for (m = 0; m < mlp->input_w->m; ++m) {
				NV_MAT_V(rand_s, 0, m) = (nv_rand() * rand_s_base) - (rand_s_base * 0.5f);
			}
#ifdef _OPENMP
#pragma omp parallel for private(y) //if (mlp->input * mlp->hidden > 10240)
#endif
			for (m = 0; m < mlp->input_w->m; ++m) {
				y = NV_MAT_V(mlp->input_bias, m, 0);
				y += nv_vector_dot(data, dm, mlp->input_w, m);
				y = nv_mlp_sigmoid(y + NV_MAT_V(rand_s, 0, m)); // y + noise

				NV_MAT_V(input_y, 0, m) = y;
			}

#ifdef _OPENMP
#pragma omp parallel for private(y) if (mlp->output > 256)
#endif
			for (m = 0; m < mlp->hidden_w->m; ++m) {
				y = NV_MAT_V(mlp->hidden_bias, m, 0);
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
				df = ir * 1.0f * NV_MAT_V(hidden_bp, 0, n);
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
				df = hr * 1.0f * NV_MAT_V(output_bp, 0, n);
				NV_MAT_V(mlp->hidden_bias, n, 0) -= df;
			}
		}
		if (nv_mlp_progress_flag) {
			printf("%d: %E, %E, %ldms\n", 
				epoch + start_epoch, data_e, sqrtf(data_e / data->m / mlp->output), nv_clock() - tm);
		}
	} while (epoch++ < max_epoch - start_epoch);

	nv_free(rand_idx);
	nv_matrix_free(&input_y);
	nv_matrix_free(&rand_s);
	nv_matrix_free(&hidden_y);
	nv_matrix_free(&output_y);
	nv_matrix_free(&hidden_bp);
}
