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
#include "nv_ml_mlp3.h"
#include <stdlib.h>

#define nv_mlp_sigmoid(a) NV_SIGMOID(a)
#define NV_MLP_MAX_EPOCH 2000
#define NV_MLP_IR 0.01f
#define NV_MLP_HR 0.0001f
#define NV_MLP_RAND_S1 1.0f
#define NV_MLP_RAND_S2 4.0f


/* 多層パーセプトロン
 * 3 Layer
 * 2Layerのほうばかりいじっていたのであっちと微妙に違うかも
 */

nv_mlp3_t *nv_mlp3_alloc(int input, int hidden1, int hidden2, int k)
{
	nv_mlp3_t *mlp = (nv_mlp3_t *)nv_malloc(sizeof(nv_mlp3_t));

	mlp->input = input;
	mlp->hidden1 = hidden1;
	mlp->hidden2 = hidden2;
	mlp->output = k;

	mlp->input_w = nv_matrix_alloc(input, hidden1);
	mlp->hidden1_w = nv_matrix_alloc(hidden1, hidden2);
	mlp->hidden2_w = nv_matrix_alloc(hidden2, k);
	mlp->input_bias = nv_matrix_alloc(1, hidden1);
	mlp->hidden1_bias = nv_matrix_alloc(1, hidden2);
	mlp->hidden2_bias = nv_matrix_alloc(1, k);

	return mlp;
}

void nv_mlp3_free(nv_mlp3_t **mlp)
{
	if (*mlp) {
		nv_matrix_free(&(*mlp)->input_w);
		nv_matrix_free(&(*mlp)->input_bias);
		nv_matrix_free(&(*mlp)->hidden1_w);
		nv_matrix_free(&(*mlp)->hidden1_bias);
		nv_matrix_free(&(*mlp)->hidden1_w);
		nv_matrix_free(&(*mlp)->hidden1_bias);
		nv_free(*mlp);
		*mlp = NULL;
	}
}


void nv_mlp3_init(nv_mlp3_t *mlp)
{
	int m, n;

	/* 初期化  */
	for (m = 0; m < mlp->input_w->m; ++m) {
		for (n = 0; n < mlp->input_w->n; ++n) {
			NV_MAT_V(mlp->input_w, m, n) = nv_rand() - 0.5f;
		}
		NV_MAT_V(mlp->input_bias, m, 0) = nv_rand() - 0.5f;
	}

	for (m = 0; m < mlp->hidden1_w->m; ++m) {
		for (n = 0; n < mlp->hidden1_w->n; ++n) {
			NV_MAT_V(mlp->hidden1_w, m, n) = nv_rand() - 0.5f;
		}
		NV_MAT_V(mlp->hidden1_bias, m, 0) = nv_rand() - 0.5f;
	}

	for (m = 0; m < mlp->hidden2_w->m; ++m) {
		for (n = 0; n < mlp->hidden2_w->n; ++n) {
			NV_MAT_V(mlp->hidden2_w, m, n) = nv_rand() - 0.5f;
		}
		NV_MAT_V(mlp->hidden2_bias, m, 0) = nv_rand() - 0.5f;
	}
}


/* クラス分類 */

int nv_mlp3_predict_label(const nv_mlp3_t *mlp, const nv_matrix_t *x, int xm)
{
	int m, n;
	int l = -1;
	float mp = -FLT_MAX, y;

	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, 1);
	nv_matrix_t *hidden1_y = nv_matrix_alloc(mlp->hidden1_w->m, 1);
	nv_matrix_t *output_y = nv_matrix_alloc(mlp->hidden2_w->m, 1);

	/* 順伝播 */
	/* 入力層 */

#ifdef _OPENMP
#pragma omp parallel for private(y) 
#endif
	for (m = 0; m < mlp->hidden1; ++m) {
		y = NV_MAT_V(mlp->input_bias, m, 0);
		y += nv_vector_dot(x, xm, mlp->input_w, m);
		NV_MAT_V(input_y, 0, m) = nv_mlp_sigmoid(y);
	}

	/* 隠れ層1 */
	for (m = 0; m < mlp->hidden2; ++m) {
		y = NV_MAT_V(mlp->hidden1_bias, m, 0);
		y += nv_vector_dot(input_y, 0, mlp->hidden1_w, m);
		NV_MAT_V(hidden1_y, 0, m) = nv_mlp_sigmoid(y);
	}

	/* 隠れ層2 出力層 */
	for (m = 0; m < mlp->output; ++m) {
		y = NV_MAT_V(mlp->hidden2_bias, m, 0);
		y += nv_vector_dot(hidden1_y, 0, mlp->hidden2_w, m);
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
	nv_matrix_free(&hidden1_y);
	nv_matrix_free(&output_y);

	return l;
}


float nv_mlp3_train_ex(nv_mlp3_t *mlp,
					  const nv_matrix_t *data,
					  const nv_matrix_t *label,
					  float ir, float hr1, float hr2,
					  int start_epoch, int end_epoch,
					  int max_epoch)
{
	float prediction;

	nv_matrix_t *t = nv_matrix_alloc(mlp->output, data->m);
	nv_mlp_make_t(t, label);
	prediction = nv_mlp3_train_lex(mlp, data, label, t, ir, hr1, hr2, start_epoch, end_epoch, max_epoch);
	nv_matrix_free(&t);

	return prediction;
}

float nv_mlp3_train_lex(nv_mlp3_t *mlp,
					 const nv_matrix_t *data,
					 const nv_matrix_t *label,
					 const nv_matrix_t *t,
					 float ir, float hr1, float hr2, 
					 int start_epoch, int end_epoch, int max_epoch)
{
	long tm;
	int m, n, im, ok;
	int epoch = 1;
	float prediction;
	float y, data_e, bp_sum;
	float org_ir = ir;
	float org_hr1 = hr1;
	float org_hr2 = hr2;
	int *rand_idx = (int *)nv_malloc(sizeof(int) * data->m);
	int label_ok;
	int do_train;

	nv_matrix_t *input_y = nv_matrix_alloc(mlp->input_w->m, 1);
	nv_matrix_t *hidden1_y = nv_matrix_alloc(mlp->hidden1_w->m, 1);
	nv_matrix_t *hidden2_y = nv_matrix_alloc(mlp->hidden2_w->m, 1);
	nv_matrix_t *output_y = nv_matrix_alloc(mlp->output, 1);
	nv_matrix_t *output_bp = nv_matrix_alloc(mlp->output, 1);
	nv_matrix_t *hidden1_bp = nv_matrix_alloc(mlp->hidden1, 1);
	nv_matrix_t *hidden2_bp = nv_matrix_alloc(mlp->hidden2, 1);
	nv_matrix_t *rand_s = nv_matrix_alloc(NV_MAX(mlp->hidden1, mlp->hidden2), 2);

	epoch = start_epoch + 1;
	do {
		if (epoch != 0 && max_epoch != 0) {
			float factor = expf(-((float)epoch * epoch) / (0.2f * max_epoch * max_epoch));
			ir = org_ir * factor;
			hr1 = org_hr1 * factor;
			hr2 = org_hr2 * factor;
		}
		
		ok = 0;
		data_e = 0.0f;
		tm = nv_clock();
		nv_shuffle_index(rand_idx, 0, data->m);
		for (im = 0; im < data->m; ++im) {
			float e;
			float mp = -FLT_MAX;
			int l = -1;
			int dm = rand_idx[im];

			/* 順伝播 */
			/* 入力層 */
			for (m = 0; m < mlp->hidden1; ++m) {
				NV_MAT_V(rand_s, 0, m) = (nv_rand() * NV_MLP_RAND_S1) - (NV_MLP_RAND_S1 * 0.5f);
			}
			for (m = 0; m < mlp->hidden2; ++m) {
				NV_MAT_V(rand_s, 1, m) = (nv_rand() * NV_MLP_RAND_S2) - (NV_MLP_RAND_S2 * 0.5f);
			}

#ifdef _OPENMP
#pragma omp parallel for private(y) //if (mlp->input * mlp->hidden > 10240)
#endif
			for (m = 0; m < mlp->input_w->m; ++m) {
				y = NV_MAT_V(mlp->input_bias, m, 0);
				y += nv_vector_dot(data, dm, mlp->input_w, m);
				y = nv_mlp_sigmoid(y + NV_MAT_V(rand_s, 0, m));
				NV_MAT_V(input_y, 0, m) = y;
			}

			/* 隠れ層1 */
#ifdef _OPENMP
#pragma omp parallel for private(y) if (mlp->output > 256)
#endif
			for (m = 0; m < mlp->hidden1_w->m; ++m) {
				y = NV_MAT_V(mlp->hidden1_bias, m, 0);
				y += nv_vector_dot(input_y, 0, mlp->hidden1_w, m);
				y = nv_mlp_sigmoid(y + NV_MAT_V(rand_s, 1, m));
				NV_MAT_V(hidden1_y, 0, m) = y;
			}

			/* 隠れ層2 */
#ifdef _OPENMP
#pragma omp parallel for private(y) if (mlp->output > 256)
#endif
			for (m = 0; m < mlp->hidden2_w->m; ++m) {
				y = NV_MAT_V(mlp->hidden2_bias, m, 0);
				y += nv_vector_dot(hidden1_y, 0, mlp->hidden2_w, m);
				NV_MAT_V(hidden2_y, 0, m) = y;
			}

			/* 出力層 */
			for (m = 0; m < mlp->hidden2_w->m; ++m) {
				NV_MAT_V(output_y, 0, m) = nv_mlp_sigmoid(NV_MAT_V(hidden2_y, 0, m));
			}

			/* 誤差 */
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
			label_ok = NV_MAT_VI(label, dm, 0);

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
			if (do_train) {
				/* 逆伝播 */
				bp_sum = 0.0f;
				for (n = 0; n < output_bp->n; ++n) {
					float y = NV_MAT_V(hidden2_y, 0, n);
					float yt = NV_MAT_V(t, dm, n);
					float expy = expf(y);
					float bp = 
						-((2.0f * yt - 1.0f) * expy + yt) 
						/ (expf(2.0f * y) + 2.0f * expy + 1.0f);
					bp_sum += bp;
					NV_MAT_V(output_bp, 0, n) = bp;
				}
				if (bp_sum != 0.0f) {
					for (m = 0; m < mlp->hidden2_w->n; ++m) {
						y = 0.0f;
						for (n = 0; n < mlp->output; ++n) {
							y += NV_MAT_V(output_bp, 0, n) * NV_MAT_V(mlp->hidden2_w, n, m);
						}
						NV_MAT_V(hidden2_bp, 0, m) = y * (1.0f - NV_MAT_V(hidden1_y, 0, m)) * NV_MAT_V(hidden1_y, 0, m);
					}
					for (m = 0; m < mlp->hidden1_w->n; ++m) {
						y = 0.0f;
						for (n = 0; n < mlp->hidden2; ++n) {
							y += NV_MAT_V(hidden2_bp, 0, n) * NV_MAT_V(mlp->hidden1_w, n, m);
						}
						NV_MAT_V(hidden1_bp, 0, m) = y * (1.0f - NV_MAT_V(input_y, 0, m)) * NV_MAT_V(input_y, 0, m);
					}

					/* I -= εΔI */
#ifdef _OPENMP
#pragma omp parallel for private(m) 
#endif
					for (n = 0; n < mlp->input_w->m; ++n) {
						for (m = 0; m < mlp->input_w->n; ++m) {
							NV_MAT_V(mlp->input_w, n, m) = 
								NV_MAT_V(mlp->input_w, n, m) 
								- ir * NV_MAT_V(data, dm, m) * NV_MAT_V(hidden1_bp, 0, n);
						}
						NV_MAT_V(mlp->input_bias, n, 0) =
							NV_MAT_V(mlp->input_bias, n, 0)
							- ir * 1.0f * NV_MAT_V(hidden1_bp, 0, n);
					}

					/* H1 -= εΔH1 */
#ifdef _OPENMP
#pragma omp parallel for private(m) 
#endif
					for (n = 0; n < mlp->hidden1_w->m; ++n) {
						for (m = 0; m < mlp->hidden1_w->n; ++m) {
							NV_MAT_V(mlp->hidden1_w, n, m) =
								NV_MAT_V(mlp->hidden1_w, n, m)
								- hr1 * NV_MAT_V(input_y, 0, m) * NV_MAT_V(hidden2_bp, 0, n);
						}
						NV_MAT_V(mlp->hidden1_bias, n, 0) =
							NV_MAT_V(mlp->hidden1_bias, n, 0)
							- hr1 * 1.0f * NV_MAT_V(hidden2_bp, 0, n);
					}
					/* H -= εΔH */
#ifdef _OPENMP
#pragma omp parallel for private(m) 
#endif
					for (n = 0; n < mlp->hidden2_w->m; ++n) {
						for (m = 0; m < mlp->hidden2_w->n; ++m) {
							NV_MAT_V(mlp->hidden2_w, n, m) =
								NV_MAT_V(mlp->hidden2_w, n, m)
								- hr2 * NV_MAT_V(hidden1_y, 0, m) * NV_MAT_V(output_bp, 0, n);

						}
						NV_MAT_V(mlp->hidden2_bias, n, 0) =
							NV_MAT_V(mlp->hidden2_bias, n, 0)
							- hr2 * 1.0f * NV_MAT_V(output_bp, 0, n);
					}
				}
			}
		}
		prediction = (float)ok / data->m;
		printf("%d: E:%E, ME:%E, %f (%d/%d), %ldms\n",
			epoch, data_e, data_e / data->m / mlp->output,
			prediction, data->m - ok,
			data->m, 
			nv_clock() - tm);
	} while (epoch++ < end_epoch);

	nv_matrix_free(&input_y);
	nv_matrix_free(&hidden1_y);
	nv_matrix_free(&hidden2_y);
	nv_matrix_free(&output_y);
	nv_matrix_free(&hidden1_bp);
	nv_matrix_free(&hidden2_bp);
	nv_matrix_free(&output_bp);
	nv_matrix_free(&rand_s);
	nv_free(rand_idx);

	return prediction;
}
