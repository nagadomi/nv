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
#include "nv_ml.h"

/* 多クラスロジスティック回帰 */
/* L1,L2正則化に対応 */

#define NV_LR_CLASS_COUNT_PENALTY_EXP 2.0f
#define NV_LR_PA_EPSILON (1.0f - 0.99f)
#define NV_LR_MARGIN     0.01f

static int nv_lr_progress_flag = 0;
void nv_lr_progress(int onoff) // 0,1,2
{
	nv_lr_progress_flag = onoff;
}

nv_lr_t *
nv_lr_alloc(int n, int k)
{
	nv_lr_t *lr = (nv_lr_t *)nv_malloc(sizeof(nv_lr_t));
	lr->n = n;
	lr->k = k;
	lr->w = nv_matrix_alloc(lr->n, k);
	nv_matrix_zero(lr->w);

	return lr;
}

void 
nv_lr_free(nv_lr_t **lr)
{
	if (*lr) {
		nv_matrix_free(&(*lr)->w);
		nv_free(*lr);
		*lr = NULL;
	}
}
nv_lr_param_t 
nv_lr_param_create(int max_epoch,
				   float grad_w,
				   nv_lr_regulaization_e reg_type, 
				   float reg_w,
				   int auto_balance
	)
{
	nv_lr_param_t param;
	param.max_epoch = max_epoch;
	param.grad_w = grad_w;
	param.reg_type = reg_type;
	param.reg_w = reg_w;
	param.auto_balance = auto_balance;
	return param;
}


static float 
nv_lr_error(const nv_matrix_t *t, int tm, const nv_matrix_t *y, int ym)
{
	float err = 0.0f;
	int n;

	for (n = 0; n < t->n; ++n) {
		float p = NV_MAT_V(y, ym, n);
		if (p > FLT_EPSILON) {
			err += NV_MAT_V(t, tm, n) * logf(p);
		}
	}
	return -err;
}

static void 
nv_lr_dw(const nv_lr_t *lr, float w,
	 nv_matrix_t *dw, int el,
	 const nv_matrix_t *data, int dj,
	 const nv_matrix_t *t, int tj,
	 const nv_matrix_t *y, int yj)
{
	int j;

	for (j = 0; j < lr->k; ++j) {
		int i;
		float y_t = NV_MAT_V(y, yj, j) - NV_MAT_V(t, tj, j);
		for (i = 0; i < lr->n; ++i) {
			NV_MAT_LIST_V(dw, el, j, i) += w * y_t * NV_MAT_V(data, dj, i);
		}
	}
}

void 
nv_lr_predict_vector(const nv_lr_t *lr, nv_matrix_t *y, int yj, 
					 const nv_matrix_t *data, int dj)
{
	int j;
	nv_matrix_t *dot = nv_matrix_alloc(lr->k, 1);
	float z = 0.0f;
	float base;

	NV_ASSERT(y->n == lr->k);

	for (j = 0; j < lr->k; ++j) {
		NV_MAT_V(dot, 0, j) = nv_vector_dot(lr->w, j, data, dj);
	}
	base = -NV_MAT_V(dot, 0, nv_vector_max_n(dot, 0));;

	for (j = 0; j < lr->k; ++j) {
		z += expf(NV_MAT_V(dot, 0, j) + base);
	}
	for (j = 0; j < lr->k; ++j) {
		NV_MAT_V(y, yj, j) = expf(NV_MAT_V(dot, 0, j) + base) / z;
	}
	nv_matrix_free(&dot);
}

void 
nv_lr_init(nv_lr_t *lr, const nv_matrix_t *data)
{
	nv_matrix_zero(lr->w);
}

void 
nv_lr_train(nv_lr_t *lr,
			const nv_matrix_t *data, const nv_matrix_t *label,
			nv_lr_param_t param)
{
	int m, n, i, j, k, l;
	long tm, tm_all = nv_clock();
	float oe = FLT_MAX, er = 1.0f, we;
	float sum_e = 0.0f;
	int epoch = 0;
	int pn = (data->m > 32) ? 32:1;
	int step = data->m / (pn);
	int threads = nv_omp_procs();
	nv_matrix_t *y = nv_matrix_alloc(lr->k, threads);
	nv_matrix_t *t = nv_matrix_alloc(lr->k, threads);
	nv_matrix_t *dw = nv_matrix_list_alloc(lr->n, lr->k, threads);
	nv_matrix_t *count = nv_matrix_alloc(lr->k, 1);
	nv_matrix_t *label_weight = nv_matrix_alloc(lr->k, 1);
	float count_max_log;
	
	nv_matrix_zero(count);
	nv_matrix_fill(label_weight, 1.0f);
	if (param.auto_balance) {
		/* クラスごとに数が違う場合に更新重みをスケーリングする */
		for (m = 0; m < data->m; ++m) {
			NV_MAT_V(count, 0, (int)NV_MAT_V(label, m, 0)) += 1.0f;
		}
		count_max_log = logf(3.0f + NV_MAT_V(count, 0, nv_vector_max_n(count, 0)));
		for (n = 0; n < count->n; ++n) {
			if (NV_MAT_V(count, 0, n) > 0.0f) {
				float count_log = logf(3.0f + NV_MAT_V(count, 0, n));
				NV_MAT_V(label_weight, 0, n) = 
					powf(count_max_log, NV_LR_CLASS_COUNT_PENALTY_EXP) 
					/ powf(count_log, NV_LR_CLASS_COUNT_PENALTY_EXP);
			} else {
				NV_MAT_V(label_weight, 0, n) = 1.0f;
			}
		}
	}
	do {
		we = 1.0f / er;
		tm = nv_clock();
		sum_e = 0.0f;

		for (m = 0; m < step; ++m) {
			nv_matrix_zero(dw);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4) reduction(+:sum_e) num_threads(threads) 
#endif
			for (i = 0; i < pn; ++i) {
				int rand_m = NV_ROUND_INT((data->m - 1) * nv_rand());
				int thread_num = nv_omp_thread_id();
				int label_i = (int)NV_MAT_V(label, rand_m, 0);
				float weight = NV_MAT_V(label_weight, 0, label_i);
				float yp;

				nv_vector_zero(t, thread_num);
				NV_MAT_V(t, thread_num, label_i) = 1.0f;
				nv_lr_predict_vector(lr, y, thread_num, data, rand_m);
				yp = NV_MAT_V(y, thread_num, (int)NV_MAT_V(label, rand_m, 0));
				
				if (yp < 1.0 - NV_LR_MARGIN) {
					nv_lr_dw(lr, weight, dw, thread_num, data, rand_m, t, thread_num, y, thread_num);
					sum_e += nv_lr_error(t, thread_num, y, thread_num);
				}
			}
			for (l = 1; l < threads; ++l) {
				for (j = 0; j < dw->m; ++j) {
					for (i = 0; i < dw->n; ++i) {
						NV_MAT_LIST_V(dw, 0, j, i) += NV_MAT_LIST_V(dw, l, j, i);
					}
				}
			}
			
#ifdef _OPENMP
#pragma omp parallel for private(n)  num_threads(threads) if (lr->k > 32)
#endif
			for (k = 0; k < lr->k; ++k) {
				switch (param.reg_type) {
                case NV_LR_REG_NONE:
					for (n = 0; n < lr->n; ++n) {
						NV_MAT_V(lr->w, k, n) -= 
							we * param.grad_w * NV_MAT_LIST_V(dw, 0, k, n);
					}
					break;
				case NV_LR_REG_L1:
					// FOBOS L1
					for (n = 0; n < lr->n; ++n) {
						NV_MAT_V(lr->w, k, n) -= 
							we * param.grad_w * NV_MAT_LIST_V(dw, 0, k, n);
					}
					for (n = 0; n < lr->n; ++n) {
						float w_i = NV_MAT_V(lr->w, k, n);
						float lambda = we * param.reg_w * (1.0f / (1.0f + epoch));
						NV_MAT_V(lr->w, k, n) = nv_sign(w_i) * NV_MAX(0.0f, (fabsf(w_i) - lambda));
					}
					break;
				case NV_LR_REG_L2:
					for (n = 0; n < lr->n; ++n) {
						NV_MAT_V(lr->w, k, n) -= 
							we * (param.grad_w * (NV_MAT_LIST_V(dw, 0, k, n)
												  + param.reg_w * NV_MAT_V(lr->w, k, n)));
					}
					break;
				}
			}
		}
		if (nv_lr_progress_flag) {
			printf("nv_lr:%d: E: %E, %ldms\n",
				epoch, sum_e / (pn * step), nv_clock() - tm);
		}
		if (nv_lr_progress_flag > 1) {
			int *ok = nv_alloc_type(int, lr->k);
			int *ng = nv_alloc_type(int, lr->k);

			memset(ok, 0, sizeof(int) * lr->k);
			memset(ng, 0, sizeof(int) * lr->k);
			for (i = 0; i < data->m; ++i) {
				int predict = nv_lr_predict_label(lr, data, i);
				int teach = (int)NV_MAT_V(label, i, 0);
				if (predict == teach) {
					++ok[teach];
				} else {
					++ng[teach];
				}
			}
			for (i = 0; i < lr->k; ++i) {
				printf("%d: ok: %d, ng: %d, %f\n", i, ok[i], ng[i], (float)ok[i] / (float)(ok[i] + ng[i]));
			}
			nv_free(ok);
			nv_free(ng);
		}
		if (nv_lr_progress_flag) {
			fflush(stdout);
		}
		if (sum_e > oe) {
			er += 1.0f;
		}
		if (er >= 50.0f) {
			break;
		}
		if (sum_e < FLT_EPSILON) {
			break;
		}
		oe = sum_e;
	} while (param.max_epoch > ++epoch);

	if (nv_lr_progress_flag) {
		printf("nv_lr: %ldms\n", nv_clock() - tm_all);
	}
	nv_matrix_free(&t);
	nv_matrix_free(&y);
	nv_matrix_free(&dw);
	nv_matrix_free(&count);
	nv_matrix_free(&label_weight);
}

float 
nv_lr_predict(const nv_lr_t *lr, 
			  const nv_matrix_t *x, int xm,
			  int k)
{
	float a, p;
	float z = 0.0f;
	int j;

	a = nv_vector_dot(lr->w, k, x, xm);
	for (j = 0; j < lr->k; ++j) {
		z += expf(nv_vector_dot(lr->w, j, x, xm) - a);
	}
	if (z == 0.0f) {
		return 0.0f;
	}
	p = 1.0f / z;

	return p;
}

int 
nv_lr_predict_label(const nv_lr_t *lr,
					const nv_matrix_t *x, int xm)
{
	int l, label = -1;
	float max_dot = -FLT_MAX;

	for (l = 0; l < lr->k; ++l) {
		float dot = nv_vector_dot(lr->w, l, x, xm);
		if (dot > max_dot) {
			max_dot = dot;
			label = l;
		}
	}
	NV_ASSERT(label != -1);

	return label;
}
#if NV_ENABLE_SSE2
static NV_INLINE __m128
nv_exp_ps(__m128 x)
{
	NV_ALIGNED(float, mm[4], 16);
	_mm_store_ps(mm, x);
	return _mm_set_ps(expf(mm[3]),expf(mm[2]), expf(mm[1]), expf(mm[0]));
}
#endif

nv_int_float_t
nv_lr_predict_label_and_probability(const nv_lr_t *lr,
									const nv_matrix_t *x, int xj)
{
#if NV_ENABLE_SSE2
	{
		nv_int_float_t ret;
		int j;
		float z = 0.0;
		nv_matrix_t *dots = nv_matrix_alloc(lr->k, 1);
		int pk_lp = (lr->k & 0xfffffffc);
		__m128 max_vec = _mm_set1_ps(-FLT_MAX);
		__m128 sumexp;
		float v_max = -FLT_MAX;	
		NV_ALIGNED(float, mm[5], 16);
		
		for (j = 0; j < pk_lp; j += 4) {
			NV_MAT_V(dots, 0, j) = nv_vector_dot(lr->w, j, x, xj);
			NV_MAT_V(dots, 0, j + 1) = nv_vector_dot(lr->w, j + 1, x, xj);
			NV_MAT_V(dots, 0, j + 2) = nv_vector_dot(lr->w, j + 2, x, xj);
			NV_MAT_V(dots, 0, j + 3) = nv_vector_dot(lr->w, j + 3, x, xj);
			max_vec = _mm_max_ps(max_vec, *(const __m128 *)&NV_MAT_V(dots, 0, j));
		}
		_mm_store_ps(mm, max_vec);
		for (j = pk_lp; j < lr->k; ++j) {
			float d = nv_vector_dot(lr->w, j, x, xj);
			if (d > v_max) {
				v_max = d;
			}
			NV_MAT_V(dots, 0, j) = d;
		}
		mm[4] = v_max;
		for (j = 0; j < 5; ++j) {
			if (mm[j] > v_max) {
				v_max = mm[j];
			}
		}
		ret.i = (int)nv_float_find_index(dots->v, 0, dots->n, v_max);
		
		max_vec = _mm_set1_ps(v_max);
		sumexp = _mm_setzero_ps();
		for (j = 0; j < pk_lp; j += 4) {
			sumexp = _mm_add_ps(
				sumexp,
				nv_exp_ps(_mm_sub_ps(*(const __m128 *)&NV_MAT_V(dots, 0, j), max_vec)));
		}
		_mm_store_ps(mm, sumexp);
		z = mm[0] + mm[1] + mm[2] + mm[3];
		for (j = pk_lp; j < lr->k; ++j) {
			z += expf(NV_MAT_V(dots, 0, j) - v_max);
		}
		ret.f = (z > 0.0f) ? 1.0f / z: 0.0f;
		
		nv_matrix_free(&dots);
		
		return ret;
	}
#else
	{
		nv_int_float_t ret;
		int j;
		float z = 0.0f;
	
		nv_matrix_t *dots = nv_matrix_alloc(lr->k, 2);
		nv_int_float_t maxv;
	
		for (j = 0; j < lr->k; ++j) {
			NV_MAT_V(dots, 0, j) = nv_vector_dot(lr->w, j, x, xj);;
		}
		maxv = nv_vector_max_ex(dots, 0);
		nv_vector_fill(dots, 1, maxv.f);
		nv_vector_sub(dots, 0, dots, 0, dots, 1);

		for (j = 0; j < lr->k; ++j) {
			z += expf(NV_MAT_V(dots, 0, j));
		}
	
		ret.i = maxv.i;
		ret.f = (z > 0.0f) ? 1.0f / z: 0.0f;
		nv_matrix_free(&dots);
		
		return ret;
	}
#endif
}

void 
nv_lr_dump_c(FILE *out, const nv_lr_t *lr, const char *name, int static_variable)
{
	char var_name[1024];

	nv_snprintf(var_name, sizeof(var_name) - 1, "%s_w", name);
	nv_matrix_dump_c(out, lr->w, var_name, 1);

	fprintf(out, "%snv_lr_t %s = {\n %d, %d, &%s\n};\n",
		static_variable ? "static ":"",
		name, lr->n, lr->k,
		var_name);
	fflush(out);
}
