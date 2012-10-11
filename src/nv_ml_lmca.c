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
#include "nv_ml_lmca.h"

/**
 * Reference:
 *  Large Margin Component Analysis 
 *  Lorenzo Torresani, Kuang-chih Lee
 */

#define NV_LMCA_RETRY_MAX 3
#define NV_LMCA_DELTA_MIN 1.0E-3f
static int nv_lmca_progress_flag = 0;

void
nv_lmca_progress(int onoff)
{
	nv_lmca_progress_flag = onoff;
}

static void
nv_lmca_init_pca(nv_matrix_t *ldm,
				 const nv_matrix_t *data)
{
	int i;
	nv_cov_t *cov;
	
	NV_ASSERT(ldm->n == data->n);
	
	cov = nv_cov_alloc(data->n);
	nv_cov_eigen_ex(cov, data, 4);
	
	for (i = 0; i < ldm->m; ++i) {
		nv_vector_copy(ldm, i, cov->eigen_vec, i);
	}
	nv_cov_free(&cov);
}

#define NV_LMCA_DHINGE(s) NV_MAX(0.0f, s)
#define NV_LMCA_HINGE(s)  NV_MAX(0.0f, s)

static void
nv_lmca_lx(nv_matrix_t *data2,
		   const nv_matrix_t *ldm,
		   const nv_matrix_t *data
	)
{
	int i;

	NV_ASSERT(data2->m == data->m && data2->n == ldm->m);
	
#ifdef _OPENMP
#pragma omp parallel for	
#endif
	for (i = 0; i < data->m; ++i) {
		nv_lmca_projection(data2, i, ldm, data, i);
	}
}

/* 目的関数 */
static float
nv_lmca_obj(const nv_matrix_t *lx,
			const nv_matrix_t *labels,
			nv_knn_result_t **eta,
			int nk,
			nv_knn_result_t **eta_lx,
			int mk,
			float margin,
			float c
	)
{
	int i, j, l;
	float eps1, eps2;
	int ok = 0;
	
	eps1 = 0.0f;
	for (i = 0; i < lx->m; ++i) {
		int il = NV_MAT_VI(labels, i, 0); 
		for (j = 0; j < nk; ++j) {
			int j_idx = eta[i][j].index;
			if (il == NV_MAT_VI(labels, j_idx, 0)) {
				eps1 += nv_euclidean2(lx, i, lx, j_idx);
			}
		}
	}
	eps2 = 0.0f;
	for (i = 0; i < lx->m; ++i) {
		int il = NV_MAT_VI(labels, i, 0); 
		for (j = 0; j < nk; ++j) {
			int j_idx = eta[i][j].index;
			if (il == NV_MAT_V(labels, j_idx, 0)) {
				float d_ij = nv_euclidean2(lx, i, lx, j_idx);
				for (l = 0; l < mk; ++l) {
					int l_idx = eta_lx[i][l].index;
					int yil = (il == NV_MAT_VI(labels, l_idx, 0)) ? 1 : 0;
					if (yil == 0) {
						float s = d_ij + margin - nv_euclidean2(lx, i, lx, l_idx);
						eps2 += NV_LMCA_HINGE(s);
					}
				}
			}
		}
	}
	for (i = 0; i < lx->m; ++i) {
		int il = NV_MAT_VI(labels, i, 0);		
		for (j = 0; j < nk; ++j) {
			int j_idx = eta_lx[i][j].index;
			ok += (il == NV_MAT_VI(labels, j_idx, 0)) ? 1 : 0;
		}
	}
	if (nv_lmca_progress_flag) {
		printf("nv_lcma: pull_error: %E push_error: %E, knn purity: %f\n", eps1, eps2, (float)ok / (lx->m * nk));
	}
	return (1.0f - c) * eps1 + c * eps2;
}

static void
nv_lmca_optimize(nv_matrix_t *ldm,
				 const nv_matrix_t *data, const nv_matrix_t *labels,
				 int nk, int mk,
				 float margin, float push_ratio, float delta,
				 int max_epoch
	)
{
	nv_matrix_t *lx = nv_matrix_alloc(ldm->m, data->m);
	nv_knn_result_t **eta = nv_alloc_type(nv_knn_result_t *, data->m);
	nv_knn_result_t **eta_lx = nv_alloc_type(nv_knn_result_t *, data->m);
	nv_matrix_t *dl, *tm1, **tm2, *ldm_old;
	int e, i;
	int procs = nv_omp_procs();
	long t = nv_clock();
	float delta_p = 1.0f;
	float tm1_scale, tm2_scale;
	float pull_ratio = 1.0f - push_ratio;
	float last_error = FLT_MAX;
	int retry_count;
	
	dl = nv_matrix_clone(ldm);
	ldm_old = nv_matrix_clone(ldm);	
	tm1 = nv_matrix_alloc(data->n, data->n);
	tm2 = nv_alloc_type(nv_matrix_t *, procs);
	for (i = 0; i < procs; ++i) {
		tm2[i] = nv_matrix_alloc(data->n, data->n);
	}
	
	/* 元の空間でのnk近傍にあるデータを保存 */
#ifdef _OPENMP
#pragma omp parallel for	
#endif
	for (i = 0; i < data->m; ++i) {
		eta[i] = nv_alloc_type(nv_knn_result_t, nk);
		nv_knn(eta[i], nk, data, data, i);

		/* 射影後の空間でのmk近傍（あとで使う) */
		eta_lx[i] = nv_alloc_type(nv_knn_result_t, mk);
	}
	
	/* 一項 */
	nv_matrix_zero(tm1);
	for (i = 0; i < data->m; ++i) {
		int j;
		int il = NV_MAT_VI(labels, i, 0);
		nv_matrix_t *d = nv_matrix_alloc(data->n, 1);
		
		for (j = 0; j < nk; ++j) {
			int j_idx = eta[i][j].index;
			if (il == NV_MAT_VI(labels, j_idx, 0)) {
				int ii, jj;
				nv_vector_sub(d, 0, data, i, data, j_idx);
				for (ii = 0; ii < d->n; ++ii) {
					for (jj = ii; jj < d->n; ++jj) {
						float v = NV_MAT_V(d, 0, jj) * NV_MAT_V(d, 0, ii);
						if (ii == jj) {
							NV_MAT_V(tm1, ii, jj) += v;
						} else {
							NV_MAT_V(tm1, ii, jj) += v;
							NV_MAT_V(tm1, jj, ii) += v;
						}
					}
				}
			}
		}
		nv_matrix_free(&d);
	}
	/*
	 * そのままだとトレードオフパラメーター(c:pull_ratio)が難しいので
	 * 絶対値の最大が1に正規化する
	 * 
	 */
	tm1_scale = 0.0f;
	for (i = 0; i < tm1->m; ++i) {
		int j;
		for (j = 0; j < tm1->n; ++j) {
			float v = fabsf(NV_MAT_V(tm1, i, j));
			if (tm1_scale < v) {
				tm1_scale = v;
			}
		}
	}
	if (tm1_scale > 0.0f) {
		tm1_scale = 1.0f / tm1_scale;
	}
	for (i = 0; i < tm1->m; ++i) {
		nv_vector_muls(tm1, i, tm1, i, tm1_scale);
	}
	if (nv_lmca_progress_flag) {
		printf("nv_lmca: tm1: %ldms\n", nv_clock() - t);
	}
	
	/* 変換行列を正則化して全データを変換する  0回目 */
	nv_vector_normalize_all(ldm);
	nv_lmca_lx(lx, ldm, data);

	/* 各データの射影後のk近傍にあるデータを保存 0回目 */
#ifdef _OPENMP
#pragma omp parallel for	
#endif
	for (i = 0; i < lx->m; ++i) {
		nv_knn(eta_lx[i], mk, lx, lx, i);
	}
	
	/* 二項 */
	for (e = 0; e < max_epoch; ++e) {
		float cur_error;
		int push_sum = 0;
		int push_count = 0;
		t = nv_clock();
		
		for (i = 0; i < procs; ++i) {
			nv_matrix_zero(tm2[i]);
		}
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (i = 0; i < data->m; ++i) {
			int j;
			const int thread_id = nv_omp_thread_id();
			const int il = NV_MAT_VI(labels, i, 0);
			nv_matrix_t *d = nv_matrix_alloc(data->n, 2);
			
			for (j = 0; j < nk; ++j) {
				const int j_idx = eta[i][j].index;
				/* nk近傍のうち同一ラベルのみについて  */
				if (il == NV_MAT_VI(labels, j_idx, 0)) {
					int l;
					int nc = 0;
					
					const float d_ij = nv_euclidean2(lx, i, lx, j_idx);
					for (l = 1; l < mk; ++l) {
						const int l_idx = eta_lx[i][l].index;
						if (il != NV_MAT_VI(labels, l_idx, 0)) {
							const float d_il = eta_lx[i][l].dist;
							const float h = NV_LMCA_DHINGE(d_ij + margin - d_il);
							if (h > 0.0f) {
								/* mk近傍のうちnk近傍内の同一ラベルからmergin以内のデータについて  */
#if NV_ENABLE_SSE
								__m128 hv = _mm_set1_ps(h);
								int ii, jj;
								int pk_lp = (data->n & 0xfffffffc);
								nv_matrix_t *mat = tm2[thread_id];
								
								nv_vector_sub(d, 0, data, i, data, j_idx);
								nv_vector_sub(d, 1, data, i, data, l_idx);
								
								for (ii = 0; ii < data->n; ++ii) {
									__m128 iv1 = _mm_set1_ps(NV_MAT_V(d, 0, ii));
									__m128 iv2 = _mm_set1_ps(NV_MAT_V(d, 1, ii));
									for (jj = 0; jj < pk_lp; jj += 4) {
										__m128 jv1 = _mm_load_ps(&NV_MAT_V(d, 0, jj));
										__m128 jv2 = _mm_load_ps(&NV_MAT_V(d, 1, jj));
										jv1 = _mm_mul_ps(iv1, jv1);
										jv2 = _mm_mul_ps(iv2, jv2);
										jv1 = _mm_sub_ps(jv1, jv2);
										jv1 = _mm_mul_ps(jv1, hv);
										jv1 = _mm_add_ps(jv1, *(const __m128 *)&NV_MAT_V(mat, ii, jj));
										_mm_store_ps(&NV_MAT_V(mat, ii, jj), jv1);
									}
									for (jj = pk_lp; jj < data->n; ++jj) {
										NV_MAT_V(mat, ii, jj) +=
											(NV_MAT_V(d, 0, ii) * NV_MAT_V(d, 0, jj)
											 -
											 NV_MAT_V(d, 1, ii) * NV_MAT_V(d, 1, jj)) * h;
									}
								}
#else
								int ii, jj;
								nv_matrix_t *mat = tm2[thread_id];
								
								nv_vector_sub(d, 0, data, i, data, j_idx);
								nv_vector_sub(d, 1, data, i, data, l_idx);

								for (ii = 0; ii < data->n; ++ii) {
									for (jj = ii; jj < data->n; ++jj) {
										const float v =
											(NV_MAT_V(d, 0, ii) *  NV_MAT_V(d, 0, jj)
											-
											 NV_MAT_V(d, 1, ii) *  NV_MAT_V(d, 1, jj)) * h;
										if (ii == jj) {
											NV_MAT_V(mat, ii, jj) += v;
										} else {
											NV_MAT_V(mat, ii, jj) += v;
											NV_MAT_V(mat, jj, ii) += v;
										}
									}
								}
#endif
								++nc;
							} else {
								/* eta_lxは距離順なのでこれより近いデータはない */
								//printf("--- break [%d] %d,%d - %f,%f\n", il, l, nc, d_ij, d_il);
								break;
							}
						}
					}
					++push_count;
					push_sum += nc;
				}
			}
			nv_matrix_free(&d);
		}
		/* 並列計算の結果を統合 */
		for (i = 1; i < procs; ++i) {
			int j;
			for (j = 0; j < tm2[0]->m; ++j) {
				nv_vector_add(tm2[0], j, tm2[0], j, tm2[i], j);
			}
		}
		/*
		 * そのままだとトレードオフパラメーター(c:pull_ratio)が難しいので
		 * 絶対値の最大が1に正規化する
		 * 
		 */
		tm2_scale = 0.0f;
		for (i = 0; i < tm2[0]->m; ++i) {
			int j;
			for (j = 0; j < tm2[0]->n; ++j) {
				float v = fabsf(NV_MAT_V(tm2[0], i, j));
				if (tm2_scale < v) {
					tm2_scale = v;
				}
			}
		}
		if (tm2_scale > 0.0f) {
			tm2_scale = 1.0f / tm2_scale;
		}
		for (i = 0; i < tm2[0]->m; ++i) {
			nv_vector_muls(tm2[0], i, tm2[0], i, tm2_scale);
		}
		/*
		 *　dL = 2 * L * tm1 + 2 * c * L * tm2
		 *
		 *  2はcに含まれるという事で省略する
		 *  cは比率でやるので
		 *   dL = (1.0 - push_ratio) * L * tm1 + push_ratio * L * tm2
		 *  を勾配とする
		 */
		/* dL = pull_ratio * L * tm1 */
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (i = 0; i < dl->n; ++i) {
			int j;
			for (j = 0; j < dl->m; ++j) {
				NV_MAT_V(dl, j, i) = pull_ratio * nv_vector_dot(ldm, j, tm1, i);
			}
		}

		/* dL += pull_ratio * L * tm2 */
#ifdef _OPENMP
#pragma omp parallel for
#endif	
		for (i = 0; i < dl->n; ++i) {
			int j;
			for (j = 0; j < dl->m; ++j) {
				NV_MAT_V(dl, j, i) += push_ratio * nv_vector_dot(ldm, j, tm2[0], i);
			}
		}
		
		/*
		 * L_new = L_old - delta * dL
		 */
		retry_count = 0;
		nv_matrix_copy_all(ldm_old, ldm);
		do {
			float w;
			if (retry_count > 0) {
				delta_p *= 0.5f;
				nv_matrix_copy_all(ldm, ldm_old);
			}
			w = delta_p * delta / sqrtf(1.0f + e);
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (i = 0; i < ldm->m; ++i) {
				int j;
				for (j = 0; j < ldm->n; ++j) {
					NV_MAT_V(ldm, i, j) -= w * NV_MAT_V(dl, i, j);
				}
			}
			/* 結果を評価する */
			
			/* 変換行列を正則化して全データを変換する  */
			nv_vector_normalize_all(ldm);
			nv_lmca_lx(lx, ldm, data);
			/* 各データの射影後のk近傍にあるデータを保存 */
#ifdef _OPENMP
#pragma omp parallel for	
#endif
			for (i = 0; i < lx->m; ++i) {
				nv_knn(eta_lx[i], mk, lx, lx, i);
			}
			
			/* 目的関数（最小化） */
			cur_error = nv_lmca_obj(lx, labels, eta, nk, eta_lx, mk, margin, push_ratio);
			retry_count += 1;
		} while (cur_error > last_error && retry_count < NV_LMCA_RETRY_MAX && delta_p > NV_LMCA_DELTA_MIN);
		if (retry_count > NV_LMCA_RETRY_MAX || delta_p < NV_LMCA_DELTA_MIN) {
			if (last_error > cur_error) {
				nv_matrix_copy_all(ldm, ldm_old);
			}
			break;
		}
		last_error = cur_error;
		if (nv_lmca_progress_flag) {
			printf("nv_lmca: %d: %ldms, loss: %f, push_avg: %d\n",
				   e, nv_clock() - t,
				   last_error,
				   push_sum / push_count
				);
		}
	}
	
	nv_matrix_free(&tm1);
	for (i = 0; i < procs; ++i) {
		nv_matrix_free(&tm2[i]);
	}
	nv_free(tm2);
	nv_matrix_free(&lx);
	nv_matrix_free(&dl);
	nv_matrix_free(&ldm_old);
	for (i = 0; i < data->m; ++i) {
		nv_free(eta[i]);
		nv_free(eta_lx[i]);
	}
	nv_free(eta);
	nv_free(eta_lx);
}

void
nv_lmca(nv_matrix_t *ldm,
		const nv_matrix_t *data, const nv_matrix_t *labels,
		int nk, int mk, float margin, float push_ratio, float delta,
		int max_epoch)
{
	NV_ASSERT(nk < mk);
	NV_ASSERT(push_ratio >= 0.0f);
	NV_ASSERT(push_ratio <= 1.0f);
	
	/* PCAで初期化 */
	nv_lmca_init_pca(ldm, data);

	/* ldmをkNNの結果がよくなるように更新 */
	nv_lmca_optimize(ldm, data, labels, nk, mk, margin, push_ratio, delta, max_epoch);
}

void
nv_lmca_projection(nv_matrix_t *y, int yj,
				   const nv_matrix_t *ldm,
				   const nv_matrix_t *x, int xj)
{
	int i;
	
	NV_ASSERT(ldm->m == y->n);
	
	for (i = 0; i < ldm->m; ++i) {
		NV_MAT_V(y, yj, i) = nv_vector_dot(ldm, i, x, xj);
	}
}
